"""SQLite work queue with leases, bounded retries and exponential backoff."""

from datetime import datetime, timedelta, timezone
import json
import threading
from uuid import uuid4

from src.store import encode, now


class JobQueue:
    def __init__(self, store, lease_seconds=300):
        self.store, self.lease_seconds = store, lease_seconds

    def enqueue(self, kind, payload, principal, idempotency_key=None):
        principal.require("ingest")
        if kind not in {"ingest", "generate_faq"}:
            raise ValueError("Unsupported job kind")
        from hashlib import sha256
        job_id = "JOB_" + (sha256((kind + idempotency_key).encode()).hexdigest() if idempotency_key else uuid4().hex)
        timestamp = now()
        with self.store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO jobs(id,kind,payload,status,available_at,created_at) VALUES(?,?,?,'QUEUED',?,?)",
                         (job_id, kind, encode(payload), timestamp, timestamp))
            self.store.audit(principal.name, "job.enqueue", job_id, {"kind": kind}, conn)
        return job_id

    def claim(self):
        timestamp = now()
        lease = (datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)).isoformat()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE jobs SET status='FAILED',error='retry budget exhausted' WHERE attempts>=max_attempts AND status IN ('QUEUED','RUNNING') AND (lease_until IS NULL OR lease_until<?)", (timestamp,))
            row = conn.execute("""SELECT * FROM jobs WHERE attempts<max_attempts AND
                ((status='QUEUED' AND available_at<=?) OR (status='RUNNING' AND lease_until<?))
                ORDER BY created_at LIMIT 1""", (timestamp, timestamp)).fetchone()
            if not row:
                return None
            token = uuid4().hex
            payload = json.loads(row["payload"])
            payload["_lease_token"] = token
            conn.execute("UPDATE jobs SET status='RUNNING',attempts=attempts+1,lease_until=?,payload=? WHERE id=?", (lease, encode(payload), row["id"]))
            return {**dict(row), "payload": payload, "attempts": row["attempts"] + 1, "lease_token": token}

    def _heartbeat(self, job, stop):
        while not stop.wait(max(.1, self.lease_seconds / 3)):
            lease = (datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)).isoformat()
            with self.store.connect() as conn:
                row = conn.execute("SELECT payload FROM jobs WHERE id=? AND status='RUNNING'", (job["id"],)).fetchone()
                if not row or json.loads(row["payload"]).get("_lease_token") != job["lease_token"]:
                    return
                conn.execute("UPDATE jobs SET lease_until=? WHERE id=?", (lease, job["id"]))

    def run_once(self, handlers):
        job = self.claim()
        if not job:
            return None
        stop = threading.Event()
        thread = threading.Thread(target=self._heartbeat, args=(job, stop), daemon=True)
        thread.start()
        error = None
        try:
            payload = {k: v for k, v in job["payload"].items() if not k.startswith("_")}
            handlers[job["kind"]](payload)
        except Exception as exc:
            error = type(exc).__name__
        finally:
            stop.set()
            thread.join(timeout=2)
        with self.store.connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if json.loads(row["payload"]).get("_lease_token") != job["lease_token"]:
                return {"job_id": job["id"], "status": "LEASE_LOST"}
            status = "SUCCEEDED" if error is None else ("FAILED" if job["attempts"] >= job["max_attempts"] else "QUEUED")
            available = (datetime.now(timezone.utc) + timedelta(seconds=2 ** job["attempts"])).isoformat()
            conn.execute("UPDATE jobs SET status=?,lease_until=NULL,error=?,available_at=? WHERE id=?", (status, error, available, job["id"]))
        return {"job_id": job["id"], "status": status, "error": error}
