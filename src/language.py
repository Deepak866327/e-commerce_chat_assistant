"""Explicit chat-language preferences; customer input never becomes prompt code."""

LANGUAGES = {"hinglish": "natural Hinglish (Hindi and English in Roman script)",
             "en": "English", "hi": "Hindi in Devanagari script"}


def language_instruction(language):
    if language not in LANGUAGES:
        raise ValueError("Choose hinglish, en or hi as the response language")
    return "Write the customer-facing answer in " + LANGUAGES[language] + "."


# Order: Hinglish, English, Hindi. These messages contain no company policy facts.
MESSAGES = {
    "followup_dispatch_status": ("Kya aapka order dispatch ho chuka hai?", "Has your order been dispatched yet?", "क्या आपका ऑर्डर भेजा जा चुका है?"),
    "followup_refund_approval": ("Kya aapko refund approve hone ki confirmation mili hai?", "Have you received confirmation that the refund was approved?", "क्या आपको रिफंड मंज़ूर होने की पुष्टि मिली है?"),
    "followup_approval_date": ("Refund approval ka update kab mila tha?", "When did you receive the refund approval update?", "रिफंड मंज़ूर होने का अपडेट आपको कब मिला था?"),
    "followup_payment_error": ("Payment karte waqt kya error message dikha tha? OTP, PIN ya card details share mat kijiye.", "What error message appeared during payment? Please don't share an OTP, PIN or card details.", "भुगतान करते समय क्या त्रुटि संदेश दिखा था? OTP, PIN या कार्ड का विवरण साझा न करें।"),
    "followup_charged_status": ("Kya is payment ke liye aapke account se paise kate hain?", "Was your account charged for this payment?", "क्या इस भुगतान के लिए आपके खाते से पैसे कटे हैं?"),
    "followup_tracking_update": ("Latest courier tracking update mein kya likha hai?", "What does the latest courier tracking update say?", "नवीनतम कूरियर ट्रैकिंग अपडेट में क्या लिखा है?"),
    "followup_expected_delivery": ("Aapko kaunsi expected delivery date dikhayi gayi thi?", "What expected delivery date were you shown?", "आपको डिलीवरी की कौन सी अनुमानित तारीख दिखाई गई थी?"),
    "followup_item_problem": ("Aap item return kyun karna chahte hain?", "What is the reason you want to return the item?", "आप सामान किस कारण वापस करना चाहते हैं?"),
    "followup_delivery_date": ("Item aapko kab deliver hua tha?", "When was the item delivered?", "सामान आपको कब मिला था?"),
    "followup_product_problem": ("Product mein kya problem aa rahi hai?", "What problem are you having with the product?", "उत्पाद में क्या समस्या आ रही है?"),
    "followup_purchase_date": ("Product aapne kab kharida tha?", "When did you purchase the product?", "आपने उत्पाद कब खरीदा था?"),
    "followup_steps_attempted": ("Is problem ko solve karne ke liye aap ab tak kya try kar chuke hain?", "What have you already tried to resolve this?", "इस समस्या को हल करने के लिए आप अब तक क्या प्रयास कर चुके हैं?"),
    "followup_blocker": ("Kis step par aapko dikkat aa rahi hai?", "Which step are you having trouble with?", "आपको किस चरण पर परेशानी हो रही है?"),
    "followup_issue_details": ("Aapki problem mein abhi kya ho raha hai?", "What is happening with the issue right now?", "आपकी समस्या में अभी क्या हो रहा है?"),
    "resolution_check": ("Kya isse aapki problem solve hui, ya abhi bhi koi dikkat hai?", "Did this resolve the issue, or are you still having trouble?", "क्या इससे आपकी समस्या हल हुई, या अब भी कोई परेशानी है?"),
    "resolution_handoff_offer": ("Main available steps se problem solve nahi kar saka. Aapne pehle human support mana kiya tha; kya ab human support ticket banana chahenge?", "I haven't been able to resolve this with the available steps. You previously declined human support; would you like a human support ticket now?", "उपलब्ध चरणों से समस्या हल नहीं हो सकी। आपने पहले मानव सहायता से मना किया था; क्या अब मानव सहायता का टिकट बनवाना चाहेंगे?"),
    "resolution_done": ("Achha, aapne confirm kiya ki problem solve ho gayi. Koi aur sawaal ho toh bataiye.", "Thanks for confirming that the issue is resolved. Let me know if you need help with anything else.", "समस्या हल होने की पुष्टि के लिए धन्यवाद। कोई और मदद चाहिए तो बताइए।"),
    "resolution_rate_limited": ("Agla message bhejne se pehle thodi der ruk jaiye.", "Please wait a moment before sending another message.", "अगला संदेश भेजने से पहले थोड़ी देर रुकिए।"),
    "resolution_handoff_pending": ("Aage ki sahayata ke liye main aapki request hamari support team tak pahunchane ki koshish kar raha hoon.", "I'll try to submit your request to our support team for further assistance.", "आगे की सहायता के लिए मैं आपका अनुरोध हमारी सहायता टीम तक पहुँचाने की कोशिश कर रहा हूँ।"),
    "resolution_handoff": ("Mujhe khed hai ki poori koshish ke baad bhi main aapki samasya ka samadhan nahi kar saka. Aage ki sahayata ke liye aapki query hamari support team ko escalate kar di gayi hai. Aapka ticket number {reference} hai.", "I'm sorry I couldn't resolve your query despite my best efforts. Your request has been escalated to our support team for further assistance. Your ticket reference is {reference}.", "मुझे खेद है कि पूरा प्रयास करने के बाद भी मैं आपकी समस्या का समाधान नहीं कर सका। आगे की सहायता के लिए आपका अनुरोध हमारी सहायता टीम को भेज दिया गया है। आपका टिकट नंबर {reference} है।"),
    "resolution_ticket_failed": ("Human support request abhi save nahi ho saki. Talk to a human se dobara try kijiye.", "The human support request could not be saved. Please try Talk to a human again.", "मानव सहायता का अनुरोध सहेजा नहीं जा सका। कृपया Talk to a human से फिर प्रयास करें।"),
    "greeting": ("Namaste! Main ShopAssist hoon. Aapko kis cheez mein madad chahiye?",
                 "Hello! I'm ShopAssist. What can I help you with?",
                 "नमस्ते! मैं ShopAssist हूँ। आपको किस चीज़ में मदद चाहिए?"),
    "small_talk": ("Main madad ke liye taiyaar hoon! Bataiye, aapko kis cheez mein help chahiye?",
                   "I'm ready to help. What do you need help with?", "मैं मदद के लिए तैयार हूँ। बताइए, आपको क्या मदद चाहिए?"),
    "thanks": ("Aapka swagat hai! Aur koi sawaal ho toh bataiye.",
               "You're welcome! Let me know if you have another question.", "आपका स्वागत है! कोई और सवाल हो तो बताइए।"),
    "farewell": ("Alvida! Jab bhi shopping support chahiye, aap pooch sakte hain.",
                 "Goodbye! Come back whenever you need shopping support.", "अलविदा! खरीदारी में मदद चाहिए तो फिर पूछिए।"),
    "capabilities": ("Main ShopAssist hoon. Main shipping, returns, refunds, cancellations aur warranty mein madad kar sakta hoon.",
                     "I'm ShopAssist. I can help with shipping, returns, refunds, cancellations and warranties.",
                     "मैं ShopAssist हूँ। मैं डिलीवरी, वापसी, रिफंड, ऑर्डर रद्द करने और वारंटी के सवालों में मदद कर सकता हूँ।"),
    "clarification": ("Zaroor. Aap apna sawaal ya problem bataiye, main madad karunga.",
                      "Please tell me what happened so I can help.", "कृपया अपनी समस्या बताइए ताकि मैं मदद कर सकूँ।"),
    "declined_handoff": ("Theek hai. Aap apni problem bataiye, main yahin madad karunga.",
                         "Okay. Tell me about the problem and I'll help here.", "ठीक है। अपनी समस्या बताइए, मैं यहीं मदद करूँगा।"),
    "human_support": ("Main human support ke liye aapki request record karne ki koshish karunga.",
                      "I'll try to record your request for human support.", "मैं मानव सहायता के लिए आपका अनुरोध दर्ज करने की कोशिश करूँगा।"),
    "support_queued": ("Aapka support ticket {reference} ban gaya hai. Request human support queue mein hai. Your support tickets mein status aur replies check kijiye.",
                       "Support ticket {reference} has been recorded in the human support queue. Check its status and replies under Your support tickets.",
                       "आपका सहायता टिकट {reference} दर्ज हो गया है। अनुरोध मानव सहायता की कतार में है। स्थिति और जवाब Your support tickets में देखें।"),
    "abstention": ("Mere paas iski verified policy information nahi hai. Aap human support team se madad le sakte hain.",
                   "I don't have verified policy information to answer this. You can ask the human support team for help.",
                   "इसका जवाब देने के लिए मेरे पास सत्यापित नीति की जानकारी नहीं है। आप मानव सहायता टीम से मदद ले सकते हैं।"),
    "out_of_scope": ("Main ShopAssist customer support assistant hoon. Main orders, shipping, returns, refunds, cancellations aur warranty mein help kar sakta hoon.",
                     "I'm ShopAssist, a shopping support assistant. I can help with orders, shipping, returns, refunds, cancellations and warranties.",
                     "मैं ShopAssist खरीदारी सहायता सहायक हूँ। मैं ऑर्डर, डिलीवरी, वापसी, रिफंड, रद्द करने और वारंटी में मदद कर सकता हूँ।"),
    "account_refund": ("Main aapka live refund status check ya refund expedite nahi kar sakta. Support team ko issue samajhne ke liye bataiye: refund approval ki koi confirmation mili hai?",
                       "I can't check or expedite your live refund. To help the support team understand the issue, have you received confirmation that the refund was approved?",
                       "मैं रिफंड की वास्तविक स्थिति जाँच या उसे तेज़ नहीं कर सकता। सहायता टीम को समस्या समझने के लिए बताइए: क्या आपको रिफंड मंज़ूर होने की पुष्टि मिली है?"),
    "account_cancel": ("Main chat se order cancel nahi kar sakta. Support team aapki cancellation request check kar sakti hai. Kya order dispatch hone ki confirmation mili hai?",
                       "I can't cancel orders from this chat. The support team can check your cancellation request. Have you received dispatch confirmation?",
                       "मैं इस चैट से ऑर्डर रद्द नहीं कर सकता। सहायता टीम आपका अनुरोध जाँच सकती है। क्या आपको ऑर्डर भेजे जाने की पुष्टि मिली है?"),
    "account_payment": ("Main aapka payment ya bank record check nahi kar sakta. Payment karte waqt kya error message dikha tha? OTP, PIN ya card details share mat kijiye.",
                        "I can't check payment or bank records. What error message appeared during payment? Please don't share an OTP, PIN or card details.",
                        "मैं भुगतान या बैंक रिकॉर्ड नहीं जाँच सकता। भुगतान करते समय क्या त्रुटि संदेश दिखा था? कृपया OTP, PIN या कार्ड का विवरण साझा न करें।"),
    "account_order": ("Main aapka live order ya courier status check nahi kar sakta. Aapke latest tracking update mein kya likha hai?",
                      "I can't check live order or courier status. What does your latest tracking update say?",
                      "मैं ऑर्डर या कूरियर की वास्तविक स्थिति नहीं जाँच सकता। आपके नवीनतम ट्रैकिंग अपडेट में क्या लिखा है?"),
    "account_followup": ("Main live account status check nahi kar sakta. Aapki di hui details support team ko issue check karne mein madad karengi. Ticket ka status aur replies Your support tickets mein check kijiye.",
                         "I can't check live account status. The details you've shared will help the support team investigate. Check the ticket status and replies under Your support tickets.",
                         "मैं खाते की वास्तविक स्थिति नहीं जाँच सकता। आपके दिए विवरण सहायता टीम को जाँच में मदद करेंगे। टिकट की स्थिति और जवाब Your support tickets में देखें।"),
    "account_generic": ("Mere paas account ya order system ka access nahi hai. Aap apni request ka detail bataiye taaki support team usse check kar sake.",
                        "I don't have access to account or order systems. Please describe your request so the support team can investigate.",
                        "मेरे पास खाते या ऑर्डर प्रणाली की पहुँच नहीं है। अपना अनुरोध बताइए ताकि सहायता टीम उसकी जाँच कर सके।"),
}


def message(key, language="hinglish", **values):
    language_instruction(language)
    return MESSAGES[key][{"hinglish": 0, "en": 1, "hi": 2}[language]].format(**values)
