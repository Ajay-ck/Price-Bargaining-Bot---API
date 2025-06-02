from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from datetime import datetime
import json
import re

app = Flask(__name__)

# Configure Google Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

class SmartBargainBot:
    def __init__(self):
        self.max_discount_percentage = 10
    
    def calculate_minimum_price(self, original_price):
        """Calculate the minimum price (90% of original price)"""
        return original_price * (1 - self.max_discount_percentage / 100)
    
    def detect_language(self, user_message):
        """Detect the language of the user's message using Gemini"""
        try:
            language_detection_prompt = f"""
Detect the language of this text and respond with ONLY the language name in English:
Text: "{user_message}"
If unsure, respond with "English". Only the language name, nothing else.
"""
            response = model.generate_content(language_detection_prompt)
            detected_language = response.text.strip().replace('"', '').replace("'", '').strip()
            return detected_language if detected_language else "English"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "English"
    
    def analyze_user_intent(self, user_message, conversation_history):
        """Analyze user's intent and current negotiation state"""
        try:
            intent_prompt = f"""
Analyze this conversation and the user's latest message to determine their intent and the current negotiation state.

Conversation History:
{self.format_conversation_history(conversation_history)}

Latest User Message: "{user_message}"

Respond with a JSON object containing:
{{
    "intent": "one of: greeting, price_inquiry, negotiation_request, acceptance, rejection, question, complaint, goodbye",
    "user_sentiment": "one of: positive, neutral, negative, frustrated",
    "price_mentioned": "any price the user mentioned or null",
    "deal_status": "one of: just_started, actively_negotiating, user_accepted, user_rejected, deal_closed",
    "negotiation_urgency": "one of: low, medium, high",
    "cultural_context": "any cultural bargaining cues or 'neutral'"
}}

Only respond with valid JSON, no other text.
"""
            
            response = model.generate_content(intent_prompt)
            try:
                intent_data = json.loads(response.text.strip())
                return intent_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "intent": "negotiation_request",
                    "user_sentiment": "neutral",
                    "price_mentioned": None,
                    "deal_status": "actively_negotiating",
                    "negotiation_urgency": "medium",
                    "cultural_context": "neutral"
                }
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return {
                "intent": "negotiation_request",
                "user_sentiment": "neutral", 
                "price_mentioned": None,
                "deal_status": "actively_negotiating",
                "negotiation_urgency": "medium",
                "cultural_context": "neutral"
            }
    
    def extract_price_from_response(self, response_text):
        """Extract price offers from bot response"""
        price_pattern = r'\$(\d+(?:\.\d{2})?)'
        matches = re.findall(price_pattern, response_text)
        if matches:
            return float(matches[-1])
        return None
    
    def create_smart_system_prompt(self, product_details, conversation_history, intent_data, user_language):
        """Create intelligent system prompt based on conversation analysis"""
        original_price = product_details['price']
        minimum_price = self.calculate_minimum_price(original_price)
        
        # Language-specific instructions
        language_instruction = ""
        if user_language.lower() != "english":
            language_instruction = f"""
LANGUAGE REQUIREMENT:
- The user is communicating in {user_language}
- You MUST respond in {user_language} language
- Keep prices in USD ($) format regardless of language
"""
        
        # Get previous bot offers to avoid repetition
        previous_offers = self.get_previous_offers(conversation_history)
        previous_offers_text = f"Previous offers made: {previous_offers}" if previous_offers else "No previous offers made yet"
        
        system_prompt = f"""
You are an intelligent multilingual bargaining agent. You must respond naturally based on the conversation context and user intent.

{language_instruction}

PRODUCT DETAILS:
- Product Name: {product_details['name']}
- Original Price: ${original_price}
- Minimum Price: ${minimum_price:.2f} (NEVER go below this)
- Maximum Discount: {self.max_discount_percentage}%

CURRENT CONVERSATION ANALYSIS:
- User Intent: {intent_data['intent']}
- User Sentiment: {intent_data['user_sentiment']}
- Deal Status: {intent_data['deal_status']}
- User's Mentioned Price: {intent_data.get('price_mentioned', 'None')}
- {previous_offers_text}

INTELLIGENT RESPONSE RULES:
1. **If user accepted a deal (intent: acceptance, deal_status: user_accepted):**
   - CLOSE THE DEAL immediately
   - Don't offer additional discounts
   - Congratulate them and finalize the purchase
   - Ask for next steps (payment, shipping, etc.)

2. **If user rejected or wants lower price:**
   - Analyze their mentioned price vs your minimum
   - If their price is above minimum, consider accepting or counter slightly higher
   - If below minimum, firmly but politely decline and offer minimum price
   - Explain value proposition

3. **If just starting (intent: greeting, price_inquiry):**
   - Welcome them warmly
   - Highlight product value
   - Offer initial small discount (2-5%) to show goodwill
   - Don't reveal maximum discount capability

4. **If actively negotiating:**
   - Respond to their specific request
   - Make strategic counter-offers
   - Don't automatically increase discount unless they push back
   - Show some resistance to make them feel they're earning the discount

5. **Pricing Strategy:**
   - Start conservative, increase only when necessary
   - If they mention a specific price, respond to THAT price
   - Don't offer better deals than what they're asking for
   - Never exceed {self.max_discount_percentage}% total discount

6. **Cultural Sensitivity:**
   - Adapt negotiation style to {user_language} cultural norms
   - Some cultures expect more back-and-forth, others prefer quick decisions

RESPONSE GUIDELINES:
- Always include a specific price in your response
- Be conversational and natural in {user_language}
- Match their energy level and urgency
- If they seem satisfied, don't oversell
- If deal_status is "user_accepted", focus on closing, not more discounts
- Make them feel smart about their negotiation

CRITICAL: Read the conversation carefully. If they've already agreed to a price, don't offer more discounts!
"""
        return system_prompt
    
    def get_previous_offers(self, conversation_history):
        """Extract previous price offers made by the bot"""
        offers = []
        for msg in conversation_history:
            if msg.get('role') == 'assistant' or msg.get('role') == 'bot':
                price = self.extract_price_from_response(msg.get('message', ''))
                if price:
                    offers.append(f"${price}")
        return offers
    
    def format_conversation_history(self, history):
        """Format conversation history"""
        if not history:
            return "This is the start of the conversation."
        
        formatted_history = ""
        for msg in history:
            role = "Customer" if msg['role'] == 'user' else "Bot"
            formatted_history += f"{role}: {msg['message']}\n"
        return formatted_history
    
    def get_latest_user_message(self, conversation_history):
        """Extract the latest user message"""
        if not conversation_history:
            return None
        
        sorted_history = sorted(conversation_history, key=lambda x: x.get('timestamp', ''))
        for msg in reversed(sorted_history):
            if msg.get('role') == 'user':
                return msg.get('message', '')
        return None
    
    def generate_response(self, product_details, conversation_history):
        """Generate intelligent bargaining response"""
        try:
            user_message = self.get_latest_user_message(conversation_history)
            
            if not user_message:
                # Initial greeting
                product_name = product_details['name']
                product_price = product_details['price']
                return (f"Hello! I see you're interested in the {product_name} listed at ${product_price}. "
                       f"It's a quality item and I'm here to help you get a great deal! What would you like to know?", 
                       None, "English", {"intent": "greeting", "deal_status": "just_started"})
            
            # Detect language and analyze intent
            detected_language = self.detect_language(user_message)
            intent_data = self.analyze_user_intent(user_message, conversation_history)
            
            # Create intelligent system prompt
            system_prompt = self.create_smart_system_prompt(
                product_details, conversation_history, intent_data, detected_language
            )
            
            conversation_context = self.format_conversation_history(conversation_history)
            
            full_prompt = f"""
{system_prompt}

CONVERSATION CONTEXT:
{conversation_context}

CUSTOMER'S LATEST MESSAGE: "{user_message}"

Respond intelligently as the bargaining agent in {detected_language}. 
Be natural and respond appropriately to their intent: {intent_data['intent']}.
If they accepted a deal, close it! If they want to negotiate, respond to their specific request.
"""
            
            response = model.generate_content(full_prompt)
            bot_response = response.text
            offered_price = self.extract_price_from_response(bot_response)
            
            return bot_response, offered_price, detected_language, intent_data
            
        except Exception as e:
            return (f"I apologize, but I'm having trouble processing your request right now. "
                   f"Please try again. Error: {str(e)}"), None, "English", {"intent": "error", "deal_status": "error"}

# Initialize the smart bargain bot
bargain_bot = SmartBargainBot()

@app.route('/api/bargain', methods=['POST'])
def bargain_chat():
    """Smart bargaining endpoint that responds naturally to user intent"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'product_details' not in data:
            return jsonify({
                'error': 'Missing required field: product_details',
                'status': 'error'
            }), 400
        
        product_details = data['product_details']
        conversation_history = data.get('conversation_history', [])
        
        # Validate product details
        if 'name' not in product_details or 'price' not in product_details:
            return jsonify({
                'error': 'Product details must include name and price',
                'status': 'error'
            }), 400
        
        # Generate intelligent response
        bot_response, offered_price, detected_language, intent_data = bargain_bot.generate_response(
            product_details, conversation_history
        )
        
        # Calculate discount information
        original_price = product_details['price']
        minimum_price = bargain_bot.calculate_minimum_price(original_price)
        
        actual_discount_percentage = 0
        if offered_price:
            actual_discount_percentage = ((original_price - offered_price) / original_price) * 100
        
        # Determine if deal is closed
        deal_closed = intent_data.get('deal_status') == 'user_accepted' or 'thank you' in bot_response.lower() and 'deal' in bot_response.lower()
        
        response_data = {
            'status': 'success',
            'bot_response': bot_response,
            'language_info': {
                'detected_language': detected_language,
                'response_language': detected_language
            },
            'negotiation_info': {
                'user_intent': intent_data.get('intent', 'unknown'),
                'deal_status': intent_data.get('deal_status', 'unknown'),
                'offered_price': offered_price,
                'discount_percentage': round(actual_discount_percentage, 2) if offered_price else 0,
                'deal_closed': deal_closed,
                'user_sentiment': intent_data.get('user_sentiment', 'neutral')
            },
            'product_info': {
                'name': product_details['name'],
                'original_price': original_price,
                'minimum_possible_price': minimum_price,
                'max_discount_percentage': bargain_bot.max_discount_percentage
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set!")
        print("Please set it using: export GEMINI_API_KEY='your_api_key_here'")
    
    print("Starting Smart Multilingual Bargaining Bot API...")
    print("Using model: gemini-1.5-flash")
    print("Maximum discount allowed: 10%")
    print("Features: Intent analysis, natural responses, smart deal closing")
    print("Single endpoint: /api/bargain")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
