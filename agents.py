# agents.py
"""
Agent definitions for the HR & Regional B2B Lead Generation system.

This module initializes the agent components used by the CrewAI framework,
defining their roles, goals, and available tools for finding companies
across the HR industry and the New England B2B sector.
It uses an LLM factory for model agnosticism.
"""

import logging
from crewai import Agent
# REMOVED: from langchain_openai import ChatOpenAI (No longer needed here)
# ADDED: Import the factory function
from utils.llm_factory import get_llm_instance
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

def initialize_agents(tools_dict):
    """
    Initialize and return all agents needed for the Lead Generation process.

    Args:
        tools_dict: Dictionary containing all tool instances

    Returns:
        Dictionary of initialized agents (research, hr_analysis, hr_reviewer,
                                         ne_b2b_analysis, ne_b2b_reviewer)
    """
    logger.info("Initializing agents for dual-stream processing...")

    # --- Get LLM instance from factory ---
    logger.info("Attempting to initialize LLM from factory...")
    try:
        agent_llm = get_llm_instance() # Call the factory function

        # Handle potential failure from the factory
        if agent_llm is None:
            logger.critical("Failed to get LLM instance from factory. Agents cannot be initialized.")
            # Raise a specific error or allow main.py to handle the fallout
            raise ValueError("LLM Initialization Failed via Factory")

        # Log the type of LLM object returned for verification
        logger.info(f"LLM Factory initialized successfully. LLM Type: {type(agent_llm).__name__}")

    except Exception as e:
        logger.error(f"Error obtaining LLM instance from factory: {e}", exc_info=True)
        raise # Re-raise the exception to stop initialization
    # --- END LLM Initialization ---


    # Initialize agents dictionary
    agents = {}

    # --- Research Agent (Unified Search) ---
    logger.info("Defining Research Agent (Unified Search)...")
    agents['research'] = Agent(
        role='Market Researcher and Company Identifier', # Broader role
        goal=( # Updated goal for both categories
            'Find online sources discussing (1) the HR industry globally/nationally AND '
            '(2) B2B companies operating primarily in New England (MA, CT, RI, VT, NH, ME). '
            'Extract company names and websites from these diverse sources.'
        ),
        backstory=( # Updated backstory
            "You are an expert market researcher specializing in identifying relevant online "
            "sources (articles, lists, directories) for specific business sectors. "
            "Your current task involves two parallel searches: one for companies serving the "
            "Human Resources sector globally, and another for general B2B companies operating "
            "within the New England region. You efficiently find diverse sources and extract "
            "company information (name, website) for further analysis."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tools_dict['web_search'], tools_dict['generic_scraper']],
        llm=agent_llm, # Pass the instance from the factory
        max_iter=Config.RESEARCH_AGENT_MAX_ITER
    )

    # --- HR-Specific Agents ---
    logger.info("Defining HR-Specific Analysis Agent...")
    agents['hr_analysis'] = Agent(
        role='HR Company Analyzer for Conference Sponsorship',
        goal=( # Focused goal for HR companies
            'Analyze companies operating within the HR sector (software, consulting, benefits, etc.) '
            'to find contact information and identify potential business challenges or opportunities '
            'that make them strong candidates for sponsoring HR conferences.'
        ),
        backstory=( # Focused backstory for HR
            "You are an expert in analyzing companies within the Human Resources industry. "
            "You understand their business models, common challenges (e.g., competition, technology adoption, "
            "regulation changes), and motivations. Your task is to research HR companies, find contacts, "
            "and pinpoint reasons why sponsoring an HR conference (for networking, lead gen, "
            "thought leadership, brand visibility within their specific industry) would be valuable for them."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tools_dict['email_finder'], tools_dict['pain_point_analyzer']],
        llm=agent_llm, # Pass the instance from the factory
        max_iter=Config.ANALYSIS_AGENT_MAX_ITER # Consider if separate limits are needed
    )

    logger.info("Defining HR-Specific Reviewer Agent...")
    agents['hr_reviewer'] = Agent(
        role='HR Lead Quality & Strategy Analyst',
        goal=( # Focused goal for HR reviews
            'Critically evaluate generated business pain points specifically for HR-industry companies. '
            'Assess their specificity, relevance to the HR sector/company niche, actionability, and '
            'direct connection to the value proposition of sponsoring an HR conference. Refine points for clarity.'
        ),
        backstory=( # Focused backstory for HR reviews
            "You are a sharp business analyst with deep expertise in the Human Resources sector. "
            "You review potential leads (HR companies) and their analyzed pain points, ensuring they are not generic "
            "but reflect genuine challenges or opportunities addressable by sponsoring an HR event "
            "(e.g., 'Need to showcase new compliance module to HR buyers', 'Struggling to differentiate from larger HR consultancies'). "
            "You ensure the strategic fit for sponsorship is clear."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[], # Relies on LLM reasoning
        llm=agent_llm, # Pass the instance from the factory
        max_iter=Config.ANALYSIS_AGENT_MAX_ITER # Reuse limit for now
    )

    # --- New England B2B Agents ---
    logger.info("Defining New England B2B Analysis Agent...")
    agents['ne_b2b_analysis'] = Agent(
        role='New England B2B Company Analyzer for HR Conference Sponsorship',
        goal=( # Focused goal for NE B2B companies
            'Analyze B2B companies operating primarily in New England to find contact information '
            'and identify potential business reasons why sponsoring a regional HR conference would be beneficial '
            '(e.g., reaching regional HR decision-makers, recruiting local talent, increasing brand visibility '
            'among NE businesses).'
        ),
        backstory=( # Focused backstory for NE B2B
            "You are an expert in analyzing regional B2B companies, particularly in New England. "
            "You understand that even non-HR companies might benefit from reaching the audience at HR conferences "
            "(HR leaders, executives from growing regional firms). Your task is to research these NE B2B companies, "
            "find contacts, and identify plausible angles for HR conference sponsorship, such as selling B2B services "
            "to HR departments, accessing regional talent pools, or general brand building within the NE business community."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tools_dict['email_finder'], tools_dict['pain_point_analyzer']], # Same tools, different context
        llm=agent_llm, # Pass the instance from the factory
        max_iter=Config.ANALYSIS_AGENT_MAX_ITER # Reuse limit for now
    )

    logger.info("Defining New England B2B Reviewer Agent...")
    agents['ne_b2b_reviewer'] = Agent(
        role='NE B2B Lead Quality & HR Conference Relevance Analyst',
        goal=( # Focused goal for NE B2B reviews
            'Critically evaluate generated business pain points for New England B2B companies. '
            'Assess their specificity, relevance to the B2B company, and crucially, the plausibility and '
            'clarity of the connection drawn to sponsoring an HR conference. Refine points to strengthen this connection.'
        ),
        backstory=( # Focused backstory for NE B2B reviews
            "You are a strategic analyst skilled at finding non-obvious connections. You review analyses "
            "of New England B2B companies, ensuring the suggested reasons for sponsoring an HR conference "
            "are logical and well-articulated (e.g., refining 'Need sales' to 'Opportunity to connect with hard-to-reach HR "
            "buyers at regional mid-sized companies for their [Service Name]' or 'Need developers' to 'Access regional tech talent pool showcased at HR conference hiring events'). "
            "You ensure the justification makes business sense."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[], # Relies on LLM reasoning
        llm=agent_llm, # Pass the instance from the factory
        max_iter=Config.ANALYSIS_AGENT_MAX_ITER # Reuse limit for now
    )

    logger.info(f"Initialized agents: {list(agents.keys())}")
    return agents
