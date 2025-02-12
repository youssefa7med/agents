import streamlit as st
from streamlit_lottie import st_lottie
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
animation = load_lottieurl('https://lottie.host/2beb66cb-6095-45fe-9f80-155888df4164/2XziOiTtfH.json')

# Create agents
finance_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Finance Tool',
    role="Get financial data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, stock_fundamentals=True)],
    markdown=True,
    instructions=[
        'Use tables to display data.',
        'Provide detailed financial analysis.',
        'Ensure accuracy in stock market insights.',
        'Include relevant stock trends and historical data when necessary.',
        'Explain complex financial terms in simple language.',
    ]
)

web_search_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Web Search Tool',
    role="Search Web",
    tools=[DuckDuckGo()],
    markdown=True,
    instructions=[
        'Always include sources.',
        'Provide the most relevant and recent financial news.',
        'Avoid speculation and prioritize factual information.',
    ]
)

team_leader_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Leader Tool',
    team=[web_search_agent, finance_agent],
    markdown=True,
    instructions=[
        'Always include sources.',
        'Use tables to display data.',
        'You are a financial assistant.',
        'Only respond to finance-related questions.',
        "If a query is not related to finance, reply: 'I specialize in financial topics. Please ask me about stock markets, investments, or financial data.'",
        'Summarize insights concisely but provide sufficient details for understanding.',
        'Use markdown formatting to make information clear and visually appealing.',
    ]
)

# Streamlit UI
st.set_page_config(page_title="Chat Bot", initial_sidebar_state='collapsed', page_icon='🤖')

st.title("📊 AI Financial & Web Search Assistant")

# Display animation
if animation:
    st_lottie(animation, speed=0.99, quality='high', height=700, width=700)

st.markdown("""
### 🔍 How It Works
1. Enter your query in the text box below.
2. Click **Get Response** to receive financial insights or web search results.
3. The AI will format responses in markdown with tables for easy readability.
""")

query = st.text_input("💬 Enter your query:")

if st.button("🚀 Get Response"):
    response = team_leader_agent.run(query)
    if response:
        st.markdown("""### 🤖 Response:""")
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.warning("No response received. Please check your query or try again.")

# Custom CSS
css_code = """
body {
  background-color: #f0f0f0; /* Light gray background */
}
"""

st.markdown(f"""<style>{css_code}</style>""", unsafe_allow_html=True)
