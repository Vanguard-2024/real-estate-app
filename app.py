from langchain_groq import ChatGroq
import streamlit as st
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import datetime

# Initialize the LLM (ChatGroq)
llm = ChatGroq(api_key="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", model="llama3-8b-8192")

# Initialize the embedding model for document and query embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure that the ChromaDB client is initialized only once
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = Client()  # Updated Client() initialization

# Get Chroma client from session state
client = st.session_state.chroma_client

# Define a new collection name with a timestamp to make it unique
new_collection_name = "real_estate_new_docs"

# Create a new collection with the unique name
collection = client.get_or_create_collection(
    name=new_collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
)
print(f"New collection '{new_collection_name}' is ready for use.")

# Function to perform Google search using the Serper API and gather real estate data
# Function to perform Google search using the Serper API and gather real estate data
def fetch_real_estate_info():
    api_key = "b4360a6824b8e1af6c15c69528fdf8269808e892"  # Replace with your Serper API key
    base_url = "https://google.serper.dev/search"
    
    # List of queries for comprehensive data
    queries = [
        "real estate market trends 2024",
        "housing market predictions 2024 INDIA",
        "current property values and mortgage rates 2024",
        "latest real estate news 2024",
    ]
    
    snippets = []  # To store fetched snippets

    for query in queries:
        try:
            # Construct the request URL
            params = {"api_key": api_key, "q": query}
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise error for non-2xx status codes

            # Parse response JSON
            data = response.json()
            snippets.extend([result.get('snippet', '') for result in data.get("organic", []) if result.get('snippet')])

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data for query '{query}': {e}")

    # Combine all snippets into a single document
    document = "\n\n".join(snippets)
    return document if snippets else None


def store_real_estate_info_in_db(additional_text=None):
    # Fetch real estate data from an external source (e.g., web scraping or API)
    real_estate_document = fetch_real_estate_info()

    # If additional text is provided, append it to the real estate document
    if real_estate_document and additional_text:
        real_estate_document += "\n\n" + additional_text  # Concatenate additional text

    # Check if real_estate_document is available
    if real_estate_document:
        document_id = f"real_estate_info_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add the document to ChromaDB
        collection.add(
            documents=[real_estate_document],
            metadatas=[{'source': 'real_estate_search', 'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}],
            ids=[document_id]
        )
        print("Real estate information successfully stored in ChromaDB.")
    else:
        print("No real estate information retrieved.")
        
# Initialize Streamlit app layout
st.title('Ask Real360')

# Initialize session state for chat messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Define domain context
domain_context = """
You are a highly knowledgeable real estate AI assistant. Your expertise spans real estate markets, property values, investment strategies, home buying and selling processes, mortgages, and other related real estate topics.Provide clear, concise, and accurate answers focused solely on real estate matters. Limit responses to the most relevant and practical information, ideally in 3-4 sentences. Avoid unnecessary elaboration, personal opinions, or off-topic content. If uncertain about the answer, be transparent and encourage the user to verify details with reliable sources. Only respond based on facts and real estate knowledge.
"""

# Function to query the vector database for relevant documents
def query_vector_db(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)  # Use the new query method
    return results['documents']  # Return all top n results as context

# Input for new user prompt
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    relevant_docs = query_vector_db(prompt)
    relevant_docs_str = [str(doc)[:1000] for doc in relevant_docs[:3]] 
    full_prompt = domain_context + " Here are some relevant documents:\n" + "\n".join(relevant_docs_str) + "\n" + prompt
    
    response = llm.invoke([HumanMessage(content=full_prompt)], max_tokens=2000, temperature=0.5)

    st.chat_message('assistant').markdown(response.content)
    st.session_state.messages.append({'role': 'assistant', 'content': response.content})

additional_info = """
Hyderabad Real Estate: Growth Continues Despite Market Shifts
Lead:"Hyderabad's real estate market shows resilience in 2024, with mixed signals emerging in recent months. Here's a comprehensive look at the city's property landscape."
Key Points:
1.Strong Overall Performance:
Total home sales of 54,483 units worth ₹33,641 crore between January-August 2024
18% year-on-year rise in registrations
41% annual jump in property values
2.Recent Market Performance:
September 2024 saw a 22% decline in registrations compared to 2023
Monthly registrations dropped from 6,304 units to 4,903 units
Registration value decreased from ₹3,459 crore to ₹2,820 crore
3.Premium Segment Growth:
Homes priced above ₹1 crore increased market share from 9% to 14%
Growing trend toward luxury housing and premium amenities
Strong demand for 3BHK configurations (64% of launches)
4.District-wise Performance:
Medchal-Malkajgiri leads with 42% market share
Rangareddy follows with 39%
Hyderabad district accounts for 19%
5.Key Growth Areas:
Western corridor remains strong due to IT sector presence
Prime locations: Gachibowli, HITEC City, Narsingi, Tellapur
Emerging areas: Bachupally, Kompally, Pragati Nagar
Expert Quote: "Hyderabad's residential market is flourishing, particularly in the luxury segment, as more homebuyers seek spacious layouts and premium amenities. This aligns with the broader national trend toward high-end housing," - Shishir Baijal, Chairman and Managing Director, Knight Frank India.
Closing: "While short-term fluctuations exist, Hyderabad's real estate market continues to demonstrate strong fundamentals and long-term growth potential, supported by infrastructure development and economic expansion."

Let me add more detailed data points for a comprehensive analysis on Hyderabad market
1.Price Trends by Micro-Markets:
Gachibowli: ₹7,200-8,500 per sq ft
HITEC City: ₹7,500-9,000 per sq ft
Kondapur: ₹6,800-7,900 per sq ft
Kukatpally: ₹5,500-6,800 per sq ft
Miyapur: ₹4,800-5,900 per sq ft
2.Commercial Real Estate Insights:
Office space absorption: 6.7 million sq ft in 2024 (YTD)
Average commercial rental: ₹65-85 per sq ft
IT/ITeS sector accounts for 71% of office space demand
Vacancy rates at 12.4%, below national average
3.Infrastructure Impact:
Metro Phase-II expansion adding 70km to existing network
SRDP (Strategic Road Development Plan) completion rate: 85%
ORR influence zones showing 15-20% premium in property rates
Airport connectivity boosting Shamshabad area rates by 25%
4.Investment Patterns:
NRI investments up by 32% year-on-year
48% of buyers are first-time homeowners
35% purchases for investment purposes
17% for upgrade/second home
5.Construction Cost Analysis:
Average construction cost: ₹2,800-3,500 per sq ft
Labor cost increased by 18% YoY
Raw material costs up by 22% YoY
Land prices appreciated by 25-30% in prime areas
6.Project Launches:
Total new launches: 42,500 units (Jan-Sept 2024)
Affordable housing (< ₹40 lakh): 28%
Mid-segment (₹40-80 lakh): 45%
Premium (₹80 lakh - 1.5 cr): 18%
Luxury (> ₹1.5 cr): 9%
7.Rental Market:
Average rental yield: 3.5-4%
2BHK average rent: ₹18,000-25,000
3BHK average rent: ₹25,000-45,000
Annual rental appreciation: 12%
8.Developer Analysis:
Top 10 developers control 45% market share
72% projects RERA registered
Average project completion time: 36 months
Customer satisfaction index: 7.8/10
9.Green Building Trends:
35% new projects with green certification
Energy efficiency features in 62% projects
Water recycling in 58% projects
Solar power integration in 42% projects
10.Future Projections:
Expected price appreciation: 8-12% annually
Estimated new supply: 55,000 units (2025)
Infrastructure investment planned: ₹28,000 crore
Employment growth projection: 15% (IT sector)
11.Demographic Insights:
Average buyer age: 35 years
65% buyers from IT/ITES sector
28% local buyers
72% migrant/NRI buyers
12.Financial Metrics:
Average home loan rate: 8.75-9.25%
Loan approval rate: 82%
Average loan tenure: 20 years
EMI to income ratio: 45%
13.INVENTORY STATUS (Q2 2024)

Total available units: 99,900
1% decrease from previous quarter
24% increase year-on-year
Inventory overhang: 17 months



REAL ESTATE NEWS ITEMS FOR HYDERABAD

[REGULATORY & POLICY UPDATES]

HEADLINE: "GHMC Introduces New Building Permission Guidelines"
• Online approval system upgraded for faster clearances
• Green building incentives announced - 10% extra FSI
• Timeline for approvals reduced to 15 days
Quote: "These reforms will streamline construction approvals significantly" - GHMC Commissioner

HEADLINE: "TS-RERA Tightens Regulations on Project Advertising"
• New guidelines for social media marketing
• Mandatory QR code linking to RERA registration
• Penalties announced for misleading advertisements

[METRO EXPANSION IMPACT]

HEADLINE: "Airport Metro Line Boosts Property Demand"
• 30% surge in enquiries along Nagole-Airport corridor
• Land prices up by 25% in Shamshabad area
• New residential projects announced along metro route
Source: CREDAI Market Analysis

[TOWNSHIP DEVELOPMENTS]

HEADLINE: "Pharma City Housing Demand Escalates"
• 5,000 acres designated for residential development
• Multiple developers acquire land parcels
• Expected employment generation: 100,000 jobs
Quote: "Pharma City will be a game-changer for East Hyderabad real estate" - Industry Expert

[COMMERCIAL REAL ESTATE]

HEADLINE: "Data Center Projects Drive Real Estate Growth"
• Microsoft announces new data center in Hyderabad
• 100-acre land acquisition in Chandanvelly
• Expected investment: ₹15,000 crores

HEADLINE: "Retail Space Demand Surges"
• New mall announcements in Kompally and Shamshabad
• Retail space absorption up by 40%
• International brands expanding presence

[EMERGING TRENDS]

HEADLINE: "Co-living Spaces Transform Rental Market"
• 15 new co-living projects announced
• Focus on IT corridor and university areas
• Average occupancy rates at 85%

HEADLINE: "Smart Home Projects Gain Traction"
• 60% of new launches include smart features
• Home automation market grows by 45%
• Premium for smart homes increases to 12-15%

[LAND ACQUISITIONS]

HEADLINE: "Government Land Auctions Set New Records"
• HMDA auctions fetch ₹2,000 crores
• Kokapet land prices reach ₹100 crores per acre
• International investors participate in bidding

[SUSTAINABILITY INITIATIVES]

HEADLINE: "Green Building Projects on the Rise"
• 30% of new projects opt for green certification
• Solar power mandatory in projects above 1000 sq.m
• Water recycling systems show 40% adoption rate

[AFFORDABLE HOUSING]

HEADLINE: "2BHK Housing Scheme Updates"
• 10,000 new units nearing completion
• Distribution process digitized
• New locations identified in peripheral areas

[MARKET INNOVATIONS]

HEADLINE: "PropTech Startups Transform Hyderabad Market"
• Virtual reality property tours become standard
• Blockchain-based property registration pilots
• AI-powered property valuation tools launched

[INDUSTRIAL REAL ESTATE]

HEADLINE: "Industrial Corridors Boost Real Estate"
• New aerospace park announced near Shamshabad
• Logistics parks development in Medchal
• Industrial land prices up by 35%

[HOSPITALITY SECTOR]

HEADLINE: "Hotels and Service Apartments Expansion"
• 5 new branded hotels announced
• Service apartment inventory to grow by 2000 units
• Focus on business traveler segment

🏢 HYDERABAD REAL ESTATE: DETAILED NEWS REPORT
[November 3, 2024]

[MAJOR DEVELOPMENT PROJECTS]

1. "KOKAPET GOLDEN MILE TRANSFORMATION"
Detailed Coverage:
• Prestige Group's "Prestige Dynasty" Launch
  - Investment: ₹2,200 crores
  - 25-acre integrated township
  - 1500 luxury apartments (3,4 BHK configurations)
  - Prices: ₹1.8 crores to ₹4.5 crores
  - Amenities include 12-acre central park, retail spaces
  - Completion date: December 2027

• DLF's "DLF Downtown" Progress
  - 3 million sq.ft IT SEZ development
  - Phase 1 (1.5 million sq.ft) nearing completion
  - Major pre-leases signed with global tech firms
  - Expected employment: 30,000 professionals

2. "FINANCIAL DISTRICT EXPANSION"
Key Developments:
• Amazon's New Campus Phase 2
  - 3 million sq.ft development
  - Investment: ₹1,800 crores
  - Employment potential: 25,000 jobs
  - Completion: Mid-2025

• "Sky View Corporate Park"
  - Mixed-use development by Salarpuria Sattva
  - 5 million sq.ft total development
  - Office, retail, and hospitality components
  - Investment: ₹2,500 crores

[INFRASTRUCTURE DEVELOPMENTS]

3. "TRANSPORT CORRIDOR UPGRADES"
Major Projects:
• Strategic Road Development Plan (SRDP)
  - Six new flyovers approved
  - Total cost: ₹3,200 crores
  - Focus areas: Gachibowli, Kukatpally, Miyapur
  - Timeline: 24 months

• Metro Extension Projects
  - Airport line construction progress: 35%
  - Property appreciation along corridor: 30-40%
  - New stations triggering micro-market development
  - Investment opportunities in last-mile connectivity

[MARKET DYNAMICS]

4. "PRICE TRENDS AND MARKET ANALYSIS"
Detailed Statistics:
• Residential Segment
  - Average price appreciation: 18% YoY
  - Gachibowli: ₹7,200/sq.ft (up 22%)
  - Kokapet: ₹8,500/sq.ft (up 25%)
  - Nanakramguda: ₹7,800/sq.ft (up 20%)

• Commercial Segment
  - Grade A office rents: ₹65-85/sq.ft/month
  - Retail spaces: ₹120-150/sq.ft/month
  - Vacancy rates down to 8%
  - Net absorption: 4.2 million sq.ft in Q3

5. "EMERGING MICRO-MARKETS"
Growth Areas:
• Tellapur-Kollur Belt
  - 15 new residential projects launched
  - Average price: ₹5,200/sq.ft
  - Infrastructure development worth ₹800 crores
  - Major developers entering the market

• Shamshabad Region
  - Airport influence zone development
  - Logistics parks spanning 200 acres
  - Residential project launches up by 40%
  - Average price appreciation: 28% YoY

[POLICY & REGULATORY UPDATES]

6. "GOVERNMENT INITIATIVES"
New Policies:
• TS-RERA Amendments
  - Stricter completion timelines
  - Enhanced buyer protection measures
  - Digital documentation platform launch
  - Penalty framework revision

• GHMC Building Regulations
  - Green building incentives
  - Height restrictions in specific zones
  - Parking norm modifications
  - Impact fee structure revision

[AFFORDABLE HOUSING SECTOR]

7. "AFFORDABLE HOUSING INITIATIVES"
Project Details:
• Government Housing Scheme
  - 50,000 units under construction
  - Locations: Jawaharnagar, Kollur, Mokila
  - Price range: ₹35-45 lakhs
  - Booking process digitization

• Private Developer Projects
  - 25 new affordable housing projects
  - Average unit size: 850-1100 sq.ft
  - Price range: ₹45-60 lakhs
  - PMAY subsidy benefits applicable

[COMMERCIAL REAL ESTATE]

8. "OFFICE SPACE DYNAMICS"
Market Updates:
• New Completions
  - Q3 2024: 3.8 million sq.ft
  - Major locations: Financial District, Hitec City
  - Pre-commitment levels: 45%
  - Average rental yield: 6.8%

• Future Supply
  - Under construction: 15 million sq.ft
  - Expected completion: 2024-25
  - Major developers: Brookfield, RMZ, Phoenix

[RETAIL SECTOR]

9. "RETAIL SPACE EVOLUTION"
New Developments:
• Mall Projects
  - Three new malls announced
  - Total area: 2.5 million sq.ft
  - Investment: ₹1,800 crores
  - Expected completion: 2025-26

• High Street Retail
  - Premium shopping corridors development
  - Jubilee Hills Road No. 36 transformation
  - Rental values up by 35%
  - International brand entries

[INVESTMENT TRENDS]

10. "INVESTMENT LANDSCAPE"
Major Deals:
• Foreign Investments
  - Blackstone acquires IT park for ₹1,800 crores
  - GIC invests ₹1,500 crores in mixed-use projects
  - Canadian Pension Fund: ₹2,000 crores commitment

• Domestic Investments
  - HDFC commits ₹1,000 crores for residential projects
  - SBI backs affordable housing with ₹800 crores
  - Local developers raise ₹2,500 crores via QIP


HEADLINE: "Amaravati Real Estate: From Setback to Surge - A Special Report"
ANCHOR INTRO: "In a remarkable turn of events, Amaravati's real estate market has witnessed an unprecedented boom following recent political developments. Our special report analyzes this dramatic transformation and what it means for investors and residents."
[MAIN STORY]
KEY HIGHLIGHTS:
1.Price Surge:50-100 percentage increase in property prices within days of election results
Land rates jumped from ₹10-15,000 to ₹40-50,000 per square yard
V-shaped recovery in market sentiment
2.Historical Context:
2019: Prices were ₹25,000-60,000 per sq yard
During YSRCP rule: Dropped to ₹9,000-18,000 per sq yard
Current rates: Back to ₹30,000-60,000 per sq yard
3.Market Dynamics:
Minimal transactions due to lack of sellers
Strong investor interest, especially in areas near institutions
High demand in zones near universities and government offices
4.Expert Analysis: [QUOTE] Prashant Thakur, Regional Director, ANAROCK: "The prices have recorded a V-shaped recovery since the election results, with land rates increasing by almost 60-100% in just days."
5.Future Outlook:
Commercial real estate sector expected to grow
Affordable housing segment likely to see major development
New project launches expected within 3-6 months
Infrastructure development to drive growth
6.Investment Zones:
Areas near educational institutions
Government office corridors
Infrastructure development zones
Commercial districts
[SPECIAL FOCUS]
Development Plans:
Revival of original master plan
Focus on trunk infrastructure
Integration of government institutions
Smart city features
[EXPERT RECOMMENDATIONS]
For Investors:
Consider long-term investment horizon
Focus on areas with infrastructure development
Watch for government policy announcements
Evaluate commercial property opportunities
[CLOSING]
"While the current surge appears promising, experts advise cautious optimism and thorough due diligence before making investment decisions. 

"""


store_real_estate_info_in_db(additional_text=additional_info)
