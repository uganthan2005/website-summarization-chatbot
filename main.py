from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any, Union
import uvicorn
import os
import uuid
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from supabase import create_client
import google.generativeai as genai
import re
from dotenv import load_dotenv

load_dotenv()

# Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



SUMMARIES_TABLE = "summary"
MAX_CONTENT_LENGTH = 100000  # Maximum content length to process
MAX_EMBEDDING_LENGTH = 2048  # Maximum length for text to embed

# Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pydantic models for request/response validation
class SummarizeRequest(BaseModel):
    user_id: str
    url: HttpUrl
    prompt: Optional[str] = "Summarize this website content"

class QuestionRequest(BaseModel):
    user_id: str
    question: str
    url: Optional[HttpUrl] = None

class CompareRequest(BaseModel):
    user_id: str
    url1: HttpUrl
    url2: Optional[HttpUrl] = None
    comparison_prompt: Optional[str] = None

class UserIdRequest(BaseModel):
    user_id: str

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Carry over the existing functions from main.py
def fetch_website_content(url: str) -> str:
    """Fetch and clean website content using BeautifulSoup"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "meta"]):
            tag.decompose()
            
        # Extract main content
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|article|post|main", re.I)) or soup.body
        
        if main_content:
            text = ' '.join(line.strip() for line in main_content.get_text().splitlines() if line.strip())
        else:
            text = ' '.join(line.strip() for line in soup.get_text().splitlines() if line.strip())
            
        # Limit content length
        return text[:MAX_CONTENT_LENGTH]
    except Exception as e:
        raise Exception(f"Error fetching content: {e}")

def get_embedding(text: str):
    """Generate embedding for the given text using Gemini"""
    try:
        # Truncate text if needed to meet embedding model constraints
        truncated_text = text[:MAX_EMBEDDING_LENGTH]
        
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=truncated_text,
            task_type="semantic_similarity"
        )
        return embedding["embedding"]
    except Exception as e:
        raise Exception(f"Embedding error: {e}")

def store_summary(user_id, url, prompt, summary, embedding):
    """Store summary and metadata in Supabase"""
    try:
        data = {
            "user_id": user_id, 
            "url": url,
            "prompt": prompt,
            "summary": summary,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }
        result = supabase.table(SUMMARIES_TABLE).insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error storing summary: {e}")
        return None
    
def get_conversation_history(user_id, limit=5):
    """Get recent conversation history for user"""
    try:
        response = supabase.table(SUMMARIES_TABLE)\
            .select("*")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Conversation history error: {e}")
        return []

def get_similar_summaries(user_id, query_embedding, limit=3):
    """Get semantically similar previous summaries using vector similarity"""
    try:
        # Use pgvector similarity search
        response = supabase.rpc(
            "match_summaries",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.7,
                "match_count": limit,
                "user_id_input": user_id
            }
        ).execute()
        
        return response.data if response.data else []
    except Exception as e:
        print(f"Similar summaries error: {e}")
        return []

def extract_user_preferences(user_id):
    """Extract user preferences from previous interactions"""
    history = get_conversation_history(user_id, limit=10)
    preferences = {
        "format_preferences": [],
        "length_preferences": [],
        "topics_of_interest": []
    }
    
    # Keywords to look for in prompts
    format_keywords = ["bullet points", "bullets", "numbered list", "summary", "outline", "key points", "highlights"]
    length_keywords = ["words", "sentences", "paragraphs", "brief", "concise", "detailed", "comprehensive"]
    
    for item in history:
        prompt = item.get("prompt", "").lower()
        
        # Check for format preferences
        for keyword in format_keywords:
            if keyword in prompt:
                preferences["format_preferences"].append(keyword)
                
        # Check for length preferences
        for keyword in length_keywords:
            if keyword in prompt:
                preferences["length_preferences"].append(keyword)
                
        # Extract word count preferences
        word_count_match = re.search(r'(\d+)\s*words', prompt)
        if word_count_match:
            preferences["length_preferences"].append(f"{word_count_match.group(1)} words")
            
        # Extract potential topics of interest from prompt
        words = re.findall(r'\b[a-z]{4,}\b', prompt)
        preferences["topics_of_interest"].extend(words)
        
    # Count occurrences and keep top preferences
    for key in preferences:
        if preferences[key]:
            preference_counts = {}
            for item in preferences[key]:
                preference_counts[item] = preference_counts.get(item, 0) + 1
            
            # Keep top 3 most frequent preferences
            preferences[key] = sorted(preference_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            preferences[key] = [item[0] for item in preferences[key]]
            
    return preferences

def build_context(user_id, current_query):
    """Build context for the AI with user history and preferences"""
    # Get conversation history
    history = get_conversation_history(user_id, limit=5)
    
    # Extract user preferences
    preferences = extract_user_preferences(user_id)
    
    # Create query embedding to find similar previous interactions
    query_embedding = get_embedding(current_query)
    similar_summaries = get_similar_summaries(user_id, query_embedding)
    
    context = "# User Context and Preferences\n\n"
    
    # Add user preferences
    context += "## User Preferences\n"
    if preferences["format_preferences"]:
        context += f"- Format preferences: {', '.join(preferences['format_preferences'])}\n"
    if preferences["length_preferences"]:
        context += f"- Length preferences: {', '.join(preferences['length_preferences'])}\n"
    
    # Add recent conversation history
    context += "\n## Recent Conversation History\n"
    for idx, item in enumerate(history[:3]):
        context += f"- Q: {item['prompt'][:100]}...\n"
        context += f"  A: {item['summary'][:150]}...\n\n"
    
    # Add similar previous interactions
    if similar_summaries:
        context += "\n## Similar Previous Interactions\n"
        for idx, item in enumerate(similar_summaries):
            context += f"- Similar Q: {item['prompt'][:100]}...\n"
            context += f"  A: {item['summary'][:150]}...\n\n"
    
    context += "\n# Your Task\n"
    context += "Based on the user's preferences and conversation history above, "
    context += "you are to respond to their current request. If they've shown preference "
    context += "for specific formats (bullet points, word counts, etc.), try to match those.\n\n"
    context += "Current request: " + current_query + "\n"
    
    return context, query_embedding

def parse_format_requirements(query):
    """Parse specific format requirements from user query"""
    format_info = {
        "word_count": None,
        "format_type": None,
        "focus_areas": []
    }
    
    # Check for word count
    word_count_match = re.search(r'(\d+)\s*words', query.lower())
    if word_count_match:
        format_info["word_count"] = int(word_count_match.group(1))
    
    # Check for format type
    if re.search(r'bullet\s*points|bullets', query.lower()):
        format_info["format_type"] = "bullet_points"
    elif re.search(r'numbered\s*list', query.lower()):
        format_info["format_type"] = "numbered_list"
    elif re.search(r'brief|concise|short', query.lower()):
        format_info["format_type"] = "brief"
    elif re.search(r'detailed|comprehensive|in-depth', query.lower()):
        format_info["format_type"] = "detailed"
    
    # Check for focus areas
    focus_match = re.search(r'focus(?:ing)?\s*on\s*([^,.]+)', query.lower())
    if focus_match:
        focus_area = focus_match.group(1).strip()
        format_info["focus_areas"].append(focus_area)
    
    return format_info

def generate_summary_instructions(format_info):
    """Generate specific instructions based on format requirements"""
    instructions = []
    
    if format_info["word_count"]:
        instructions.append(f"Generate a summary of approximately {format_info['word_count']} words.")
    
    if format_info["format_type"] == "bullet_points":
        instructions.append("Format the summary as bullet points highlighting key information.")
    elif format_info["format_type"] == "numbered_list":
        instructions.append("Format the summary as a numbered list of main points.")
    elif format_info["format_type"] == "brief":
        instructions.append("Keep the summary brief and to the point.")
    elif format_info["format_type"] == "detailed":
        instructions.append("Provide a detailed summary with comprehensive coverage.")
    
    if format_info["focus_areas"]:
        areas = ", ".join(format_info["focus_areas"])
        instructions.append(f"Focus particularly on aspects related to: {areas}")
    
    return " ".join(instructions)

def summarize_website(user_id, url, prompt):
    """Summarize website content based on user preferences"""
    try:
        # Fetch website content
        content = fetch_website_content(url)
        
        # Build context with user history and preferences
        context, query_embedding = build_context(user_id, prompt)
        
        # Parse format requirements
        format_info = parse_format_requirements(prompt)
        format_instructions = generate_summary_instructions(format_info)
        
        # Build the full prompt for the AI
        full_prompt = f"{context}\n\n"
        if format_instructions:
            full_prompt += f"Format instructions: {format_instructions}\n\n"
        full_prompt += f"Content to summarize:\n{content}\n\n"
        full_prompt += "Generate a summary that matches the user's preferences and request."
        
        # Generate summary using Gemini
        config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=config)
        chat = model.start_chat()
        result = chat.send_message(full_prompt)
        summary = result.text
        
        # Store the summary in Supabase
        stored_data = store_summary(user_id, url, prompt, summary, query_embedding)
        
        return {
            "success": True,
            "summary": summary,
            "stored_id": stored_data.get("id") if stored_data else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def answer_question(user_id, question, url=None):
    """Answer user questions based on current and previous context"""
    try:
        # Build context with user history and preferences
        context, query_embedding = build_context(user_id, question)
        
        # Fetch website content if URL is provided
        content = ""
        if url:
            content = fetch_website_content(url)
        
        # If no URL provided, rely more on previous summaries
        if not url or not content:
            # Get more history to work with
            history = get_conversation_history(user_id, limit=10)
            
            # Add relevant content from history
            history_content = []
            for item in history:
                history_content.append(f"URL: {item['url']}\nPrompt: {item['prompt']}\nSummary: {item['summary']}")
            
            content = "\n\n".join(history_content)
        
        # Parse format requirements
        format_info = parse_format_requirements(question)
        format_instructions = generate_summary_instructions(format_info)
        
        # Build the full prompt for the AI
        full_prompt = f"{context}\n\n"
        if format_instructions:
            full_prompt += f"Format instructions: {format_instructions}\n\n"
        
        full_prompt += f"Question: {question}\n\n"
        
        if content:
            full_prompt += f"Relevant Content:\n{content}\n\n"
            
        full_prompt += "Generate a helpful answer based on the user's question and their preferences from past interactions."
        
        # Generate answer using Gemini
        config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=config)
        chat = model.start_chat()
        result = chat.send_message(full_prompt)
        answer = result.text
        
        # Store the answer in Supabase
        stored_data = store_summary(user_id, url, question, answer, query_embedding)
        
        return {
            "success": True,
            "answer": answer,
            "stored_id": stored_data.get("id") if stored_data else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def compare_summaries(user_id, url1, url2=None, comparison_prompt=None):
    """Compare two website summaries or current website with previous summary"""
    try:
        # Get first website content
        content1 = fetch_website_content(url1)
        
        content2 = ""
        if url2:
            # If second URL provided, get its content
            content2 = fetch_website_content(url2)
        else:
            # If no second URL, find most relevant previous summary
            query_embedding = get_embedding(content1[:MAX_EMBEDDING_LENGTH])
            similar_summaries = get_similar_summaries(user_id, query_embedding, limit=1)
            
            if similar_summaries:
                content2 = similar_summaries[0].get("summary", "")
                url2 = similar_summaries[0].get("url", "Previous summary")
        
        # Build context
        prompt = comparison_prompt or f"Compare the content from {url1} with {url2}"
        context, query_embedding = build_context(user_id, prompt)
        
        # Build the full prompt for comparison
        full_prompt = f"{context}\n\n"
        full_prompt += f"Content 1 ({url1}):\n{content1[:MAX_CONTENT_LENGTH//2]}\n\n"
        full_prompt += f"Content 2 ({url2 or 'Previous summary'}):\n{content2[:MAX_CONTENT_LENGTH//2]}\n\n"
        full_prompt += "Compare these two contents highlighting key similarities and differences. "
        full_prompt += "Focus on main themes, arguments, and conclusions."
        
        # Generate comparison using Gemini
        config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=config)
        chat = model.start_chat()
        result = chat.send_message(full_prompt)
        comparison = result.text
        
        # Store the comparison in Supabase
        stored_data = store_summary(user_id, f"{url1}, {url2}", prompt, comparison, query_embedding)
        
        return {
            "success": True,
            "comparison": comparison,
            "stored_id": stored_data.get("id") if stored_data else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_user_analytics(user_id):
    """Get analytics about user's summarization patterns"""
    try:
        # Get user's entire history
        history = get_conversation_history(user_id, limit=100)
        
        if not history:
            return {
                "success": False,
                "error": "No history found for this user"
            }
            
        analytics = {
            "total_summaries": len(history),
            "frequent_domains": {},
            "average_summary_length": 0,
            "common_topics": [],
            "usage_over_time": {}
        }
        
        total_length = 0
        all_prompts = ""
        
        for item in history:
            # Extract domain from URL
            url = item.get("url", "")
            if url:
                try:
                    domain = url.split("//")[-1].split("/")[0]
                    analytics["frequent_domains"][domain] = analytics["frequent_domains"].get(domain, 0) + 1
                except:
                    pass
                    
            # Calculate summary length
            summary = item.get("summary", "")
            total_length += len(summary.split())
            
            # Collect prompts for topic analysis
            all_prompts += item.get("prompt", "") + " "
            
            # Track usage over time
            timestamp = item.get("timestamp", "")
            if timestamp:
                try:
                    date = timestamp.split("T")[0]
                    analytics["usage_over_time"][date] = analytics["usage_over_time"].get(date, 0) + 1
                except:
                    pass
                    
        # Calculate average summary length
        if analytics["total_summaries"] > 0:
            analytics["average_summary_length"] = total_length / analytics["total_summaries"]
            
        # Sort domains by frequency
        analytics["frequent_domains"] = sorted(
            analytics["frequent_domains"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Extract common topics using Gemini
        if all_prompts:
            topic_prompt = f"Analyze these user requests and extract the 5 most common topics or themes:\n{all_prompts[:5000]}"
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(topic_prompt)
            if response.text:
                topics = response.text
                analytics["common_topics"] = topics
                
        # Sort usage over time
        analytics["usage_over_time"] = dict(sorted(analytics["usage_over_time"].items()))
        
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Create FastAPI app
app = FastAPI(title="Website Summarizer API",description="API for summarizing website content, answering questions, and comparing websites",version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Website Summarizer API", "status": "online"}

@app.post("/summarize", response_model=ApiResponse, tags=["Summarization"])
async def api_summarize_website(request: SummarizeRequest, background_tasks: BackgroundTasks):
    try:
        result = summarize_website(request.user_id, request.url, request.prompt)
        if result["success"]:
            return {
                "success": True,
                "data": {
                    "summary": result["summary"],
                    "stored_id": result["stored_id"]
                }
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/question", response_model=ApiResponse, tags=["Questions"])
async def api_answer_question(request: QuestionRequest):
    try:
        result = answer_question(request.user_id, request.question, request.url)
        if result["success"]:
            return {
                "success": True,
                "data": {
                    "answer": result["answer"],
                    "stored_id": result["stored_id"]
                }
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/compare", response_model=ApiResponse, tags=["Comparison"])
async def api_compare_websites(request: CompareRequest):
    try:
        result = compare_summaries(request.user_id, request.url1, request.url2, request.comparison_prompt)
        if result["success"]:
            return {
                "success": True,
                "data": {
                    "comparison": result["comparison"],
                    "stored_id": result["stored_id"]
                }
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/analytics", response_model=ApiResponse, tags=["Analytics"])
async def api_user_analytics(request: UserIdRequest):
    try:
        result = get_user_analytics(request.user_id)
        if result["success"]:
            return {
                "success": True,
                "data": {
                    "analytics": result["analytics"]
                }
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/history/{user_id}", response_model=ApiResponse, tags=["History"])
async def api_get_history(user_id: str, limit: int = Query(5, ge=1, le=100)):
    try:
        history = get_conversation_history(user_id, limit)
        return {
            "success": True,
            "data": {
                "history": history
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Add schemas for the Swagger documentation
@app.get("/schemas", tags=["Documentation"])
async def get_schemas():
    schemas = {
        "SummarizeRequest": {
            "user_id": "string",
            "url": "string (valid URL)",
            "prompt": "string (optional)"
        },
        "QuestionRequest": {
            "user_id": "string",
            "question": "string",
            "url": "string (valid URL, optional)"
        },
        "CompareRequest": {
            "user_id": "string",
            "url1": "string (valid URL)",
            "url2": "string (valid URL, optional)",
            "comparison_prompt": "string (optional)"
        },
        "UserIdRequest": {
            "user_id": "string"
        }
    }
    return schemas

