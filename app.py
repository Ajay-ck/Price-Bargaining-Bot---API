from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from datetime import datetime
import json
from flask_cors import CORS
import re

app = Flask(__name__)

# Configure Google Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Use the current stable Gemini model
# Options: 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17'
model = genai.GenerativeModel('gemini-1.5-flash')

class BargainBot:
    def __init__(self):
        self.max_discount_percentage = 10
        self.negotiation_steps = {
            1: {"discount": 2, "description": "Initial small discount"},
            2: {"discount": 4, "description": "Second offer - showing flexibility"},
            3: {"discount": 7, "description": "Third offer - getting serious"},
            4: {"discount": 10, "description": "Final offer - maximum discount"}
        }
    
    def calculate_minimum_price(self, original_price):
        """Calculate the minimum price (90% of original price)"""
        return original_price * (1 - self.max_discount_percentage / 100)
    
    def get_negotiation_step(self, conversation_history):
        """Determine which step of negotiation we're in based on conversation history"""
        if not conversation_history:
            return 1
        
        # Count user messages to determine negotiation step
        user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
        step = len(user_messages)
        
        # Cap at 4 steps maximum
        return min(step, 4)
    
    def detect_language(self, user_message):
        """Detect the language of the user's message using Gemini"""
        try:
            language_detection_prompt = f"""
Detect the language of this text and respond with ONLY the language name in English (e.g., "Spanish", "French", "Arabic", "Hindi", "Chinese", "Japanese", "German", etc.):

Text: "{user_message}"

If the text is in English or you cannot determine the language, respond with "English".
Respond with only the language name, nothing else.
"""
            
            response = model.generate_content(language_detection_prompt)
            detected_language = response.text.strip()
            
            # Clean up the response to get just the language name
            detected_language = detected_language.replace('"', '').replace("'", '').strip()
            
            return detected_language if detected_language else "English"
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "English"  # Default to English if detection fails
    
    def extract_price_from_response(self, response_text):
        """Extract price offers from bot response"""
        # Look for patterns like $XX.XX or $XXX
        price_pattern = r'\$(\d+(?:\.\d{2})?)'
        matches = re.findall(price_pattern, response_text)
        if matches:
            return float(matches[-1])  # Return the last price mentioned
        return None
    
    def create_system_prompt(self, product_details, conversation_history, negotiation_step, user_language):
        """Create system prompt for the bargaining bot with language support"""
        original_price = product_details['price']
        minimum_price = self.calculate_minimum_price(original_price)
        
        step_info = self.negotiation_steps.get(negotiation_step, self.negotiation_steps[4])
        suggested_discount = step_info["discount"]
        suggested_price = original_price * (1 - suggested_discount / 100)
        
        # Language-specific instructions
        language_instruction = ""
        if user_language.lower() != "english":
            language_instruction = f"""
LANGUAGE REQUIREMENT:
- The user is communicating in {user_language}
- You MUST respond in {user_language} language
- Maintain natural, fluent conversation in {user_language}
- All prices should still be in USD ($) format regardless of language
- Keep the same professional bargaining tone in {user_language}
"""
        
        system_prompt = f"""
You are a skilled multilingual bargaining agent for an e-commerce platform. Your role is to negotiate with customers while protecting business interests.

{language_instruction}

PRODUCT DETAILS:
- Product Name: {product_details['name']}
- Original Price: ${original_price}
- Category: {product_details.get('category', 'General')}
- Description: {product_details.get('description', 'Quality product')}

BARGAINING RULES AND CONSTRAINTS:
1. MAXIMUM discount allowed: {self.max_discount_percentage}% (Minimum price: ${minimum_price:.2f})
2. NEVER exceed {self.max_discount_percentage}% discount under any circumstances
3. Current negotiation step: {negotiation_step}/4
4. For this step, offer around {suggested_discount}% discount (approximately ${suggested_price:.2f})
5. Be strategic - don't give maximum discount immediately
6. Make customer feel they're earning each discount through negotiation

NEGOTIATION STRATEGY BY STEP:
- Step 1: Small discount (2-3%) - Test their interest
- Step 2: Moderate discount (4-5%) - Show flexibility  
- Step 3: Significant discount (6-8%) - Getting serious
- Step 4: Final offer (9-10%) - Maximum possible discount

RESPONSE GUIDELINES:
1. Always mention a specific price offer in your response
2. Justify the discount with product value, quality, or limited-time reasoning
3. If customer asks for more than {self.max_discount_percentage}%, politely decline and offer alternatives
4. Use phrases appropriate for {user_language} like "I can go as low as..." or "My best offer is..."
5. Be friendly but firm about your limits
6. If this is step 4, make it clear this is your final offer
7. Adapt your language to match the product type and price range
8. For higher-priced items, emphasize value and quality
9. For lower-priced items, focus on affordability and deals
10. Maintain cultural sensitivity and appropriate business etiquette for {user_language}

IMPORTANT REMINDERS:
- Never reveal the minimum price directly or that you have a {self.max_discount_percentage}% limit
- Always respond in {user_language} if it's not English
- Keep prices in USD format ($XX.XX) regardless of language
- Be culturally appropriate in your negotiation style for {user_language} speakers
"""
        return system_prompt
    
    def format_conversation_history(self, history):
        """Format conversation history for the prompt"""
        if not history:
            return "This is the start of the conversation."
        
        formatted_history = "Previous conversation:\n"
        for msg in history:
            role = "Customer" if msg['role'] == 'user' else "Bot"
            formatted_history += f"{role}: {msg['message']}\n"
        
        return formatted_history
    
    def get_latest_user_message(self, conversation_history):
        """Extract the latest user message from conversation history"""
        if not conversation_history:
            return None
            
        # Sort by timestamp to get the latest message
        sorted_history = sorted(conversation_history, key=lambda x: x.get('timestamp', ''))
        
        # Find the latest user message
        for msg in reversed(sorted_history):
            if msg.get('role') == 'user':
                return msg.get('message', '')
        
        return None
    
    def generate_response(self, product_details, conversation_history):
        """Generate bargaining response using Gemini with language detection"""
        try:
            # Get the latest user message
            user_message = self.get_latest_user_message(conversation_history)
            
            if not user_message:
                return "I'm here to help you get the best deal! What would you like to know about this item?", None, 1, "English"
            
            # Detect the language of the user's message
            detected_language = self.detect_language(user_message)
            print(f"Detected language: {detected_language}")  # For debugging
            
            # Determine negotiation step
            negotiation_step = self.get_negotiation_step(conversation_history)
            
            system_prompt = self.create_system_prompt(product_details, conversation_history, negotiation_step, detected_language)
            conversation_context = self.format_conversation_history(conversation_history)
            
            full_prompt = f"""
{system_prompt}

{conversation_context}

Customer's latest message: "{user_message}"

Respond as the bargaining agent in {detected_language} language. Make a specific price offer based on the negotiation step. Be conversational and strategic.
"""
            
            response = model.generate_content(full_prompt)
            bot_response = response.text
            
            # Extract offered price from response
            offered_price = self.extract_price_from_response(bot_response)
            
            return bot_response, offered_price, negotiation_step, detected_language
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request right now. Please try again. Error: {str(e)}", None, 1, "English"

# Initialize the bargain bot
bargain_bot = BargainBot()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Universal Multilingual Bargaining Chatbot API',
        'status': 'running',
        'description': 'Negotiate prices for any product in multiple languages',
        'model': 'gemini-1.5-flash',
        'features': [
            'Multi-language support',
            'Automatic language detection',
            'Cultural negotiation adaptation',
            'Price negotiation in any language'
        ],
        'available_endpoints': {
            'health_check': 'GET /health',
            'initial_message': 'POST /api/bargain/initial',
            'bargain_chat': 'POST /api/bargain',
            'product_info': 'POST /api/bargain/product-info'
        },
        'max_discount': f"{bargain_bot.max_discount_percentage}%",
        'supported_languages': 'Auto-detected (Spanish, French, Arabic, Hindi, Chinese, Japanese, German, and more)',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/bargain', methods=['POST'])
def bargain_chat():
    """Main bargaining endpoint with language support"""
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
        
        # Generate response based on conversation history with language detection
        bot_response, offered_price, negotiation_step, detected_language = bargain_bot.generate_response(
            product_details, 
            conversation_history
        )
        
        # Calculate current minimum price and discount info
        original_price = product_details['price']
        minimum_price = bargain_bot.calculate_minimum_price(original_price)
        
        # Calculate actual discount if price was offered
        actual_discount_percentage = 0
        if offered_price:
            actual_discount_percentage = ((original_price - offered_price) / original_price) * 100
        
        response_data = {
            'status': 'success',
            'bot_response': bot_response,
            'language_info': {
                'detected_language': detected_language,
                'response_language': detected_language
            },
            'negotiation_info': {
                'step': negotiation_step,
                'max_steps': 4,
                'offered_price': offered_price,
                'discount_percentage': round(actual_discount_percentage, 2) if offered_price else 0,
                'is_final_offer': negotiation_step >= 4
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

@app.route('/api/bargain/initial', methods=['POST'])
def get_initial_message():
    """Get initial welcoming message for any product with language detection"""
    try:
        data = request.get_json()
        
        if 'product_details' not in data:
            return jsonify({
                'error': 'Missing product_details',
                'status': 'error'
            }), 400
        
        product_details = data['product_details']
        
        if 'name' not in product_details or 'price' not in product_details:
            return jsonify({
                'error': 'Product details must include name and price',
                'status': 'error'
            }), 400
        
        # Get user's preferred language if provided
        user_language = data.get('language', 'English')
        
        # Generic initial welcoming message
        product_name = product_details['name']
        product_price = product_details['price']
        
        # Create contextual greeting based on price range
        if product_price < 50:
            price_context = "affordable"
        elif product_price < 200:
            price_context = "reasonably priced"
        elif product_price < 500:
            price_context = "premium"
        else:
            price_context = "high-end"
        
        # Generate initial message in specified language
        if user_language.lower() != "english":
            language_prompt = f"""
Generate a welcoming initial message for a bargaining chatbot in {user_language} language.

Product: {product_name}
Price: ${product_price}
Context: {price_context} item

The message should:
1. Greet the customer warmly in {user_language}
2. Acknowledge their interest in the product
3. Mention the price in USD format
4. Offer to help them get the best deal
5. Be culturally appropriate for {user_language} speakers

Keep it friendly and professional in {user_language}.
"""
            
            try:
                response = model.generate_content(language_prompt)
                initial_message = response.text.strip()
            except Exception as e:
                # Fallback to English if translation fails
                initial_message = f"Hello! I see you're interested in the {product_name} listed at ${product_price}. It's a {price_context} item and I'm here to help you get the best possible deal! What would you like to discuss about this product?"
                user_language = "English"
        else:
            initial_message = f"Hello! I see you're interested in the {product_name} listed at ${product_price}. It's a {price_context} item and I'm here to help you get the best possible deal! What would you like to discuss about this product?"
        
        # Calculate discount info
        minimum_price = bargain_bot.calculate_minimum_price(product_details['price'])
        max_discount_amount = product_details['price'] - minimum_price
        
        return jsonify({
            'status': 'success',
            'initial_message': initial_message,
            'language_info': {
                'response_language': user_language
            },
            'product_info': {
                'name': product_details['name'],
                'original_price': product_details['price'],
                'potential_savings': f"Up to ${max_discount_amount:.2f} possible"
            },
            'negotiation_info': {
                'max_discount_percentage': bargain_bot.max_discount_percentage,
                'total_steps': 4,
                'strategy': 'Step-by-step negotiation'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Universal Multilingual Bargaining Chatbot API',
        'description': 'Ready to negotiate prices for any product in multiple languages',
        'model': 'gemini-1.5-flash',
        'features': [
            'Automatic language detection',
            'Multi-language responses',
            'Cultural adaptation',
            'Price negotiation'
        ],
        'max_discount': f"{bargain_bot.max_discount_percentage}%",
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Make sure to set your GEMINI_API_KEY environment variable
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set!")
        print("Please set it using: export GEMINI_API_KEY='your_api_key_here'")
    
    print("Starting Universal Multilingual Bargaining Bot API...")
    print("Using model: gemini-1.5-flash")
    print("Maximum discount allowed: 10%")
    print("Negotiation strategy: 4-step process")
    print("Language support: Auto-detection and multi-language responses")
    print("Ready to handle any product type and price range in multiple languages")
    
    app.run(debug=True, host='0.0.0.0', port=5000)