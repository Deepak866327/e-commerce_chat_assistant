"""Explicit chat-language preferences; customer input never becomes prompt code."""

LANGUAGES = {"hinglish": "natural Hinglish (Hindi and English in Roman script)",
             "en": "English", "hi": "Hindi in Devanagari script"}


def language_instruction(language):
    if language not in LANGUAGES:
        raise ValueError("Choose hinglish, en or hi as the response language")
    return "Write the customer-facing answer in " + LANGUAGES[language] + "."


# Order: Hinglish, English, Hindi. These messages contain no company policy facts.
MESSAGES = {
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
    "account_generic": ("Mere paas account ya order system ka access nahi hai. Aap apni request ka detail bataiye taaki support team usse check kar sake.",
                        "I don't have access to account or order systems. Please describe your request so the support team can investigate.",
                        "मेरे पास खाते या ऑर्डर प्रणाली की पहुँच नहीं है। अपना अनुरोध बताइए ताकि सहायता टीम उसकी जाँच कर सके।"),
}


def message(key, language="hinglish", **values):
    language_instruction(language)
    return MESSAGES[key][{"hinglish": 0, "en": 1, "hi": 2}[language]].format(**values)
