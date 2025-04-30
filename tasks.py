# tasks.py
import logging
from crewai import Task, Agent # Import Agent for type checking and passing agent instances

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Already done in main
logger = logging.getLogger(__name__)

# --- SEARCH TASKS (Unified for Research Agent) ---

def create_search_tasks(agents):
    """
    Create tasks for the Research Agent to find sources for BOTH
    HR industry companies AND New England regional B2B companies.
    """
    logger.info("Creating unified search tasks for HR & NE B2B sources...")

    plan_search_task = None
    execute_search_task = None
    research_agent = agents.get('research') # Should contain the single Research Agent

    if not isinstance(research_agent, Agent):
        logger.error("Research agent not found or invalid in create_search_tasks. Cannot create search tasks.")
        return [] # Return empty list if agent is invalid

    try:
        plan_search_task = Task(
            description=(
                # UPDATED Description for dual search
                "Develop **two distinct sets** of search queries (3-4 queries per set) within a single plan. "
                "**Set 1:** Target sources discussing companies in the **HR industry** (software, consulting, recruiting, benefits, payroll, training, compliance, DEI, etc.). Aim for sources likely to list multiple HR-focused companies nationally or globally. "
                "**Set 2:** Target sources (regional business journals, directories, award lists, industry associations in NE) discussing **B2B companies headquartered or primarily operating in New England** (MA, CT, RI, VT, NH, ME). Focus on sources likely to list multiple B2B service or product providers in the region. "
                "The final output should combine queries from both sets into one list."
            ),
#             expected_output=(
#                # UPDATED Expected Output
#                "A single Python list of 6-8 relevant query strings, covering both HR industry sources and New England B2B sources."
#            ),
            # --- MODIFIED: Make expected output MUCH more explicit ---
            expected_output=(
                "**CRITICAL:** Your final output MUST be ONLY a Python list of strings, where each string is a unique URL. "
                "Example format: ['https://example.com/list1', 'https://anothersite.org/article', 'https://regionalsource.net/directory']\n"
                "Do NOT include any introductory text, concluding remarks, notes, or any other text before or after the Python list itself. "
                "The output should start directly with '[' and end directly with ']'. Provide up to 10 unique URLs."
            ),
            # --- END MODIFICATION ---


            agent=research_agent # Use the validated single research agent
        )

        execute_search_task = Task(
            description=(
                # UPDATED Description for dual search execution
                "Take the combined list of HR industry and NE B2B focused queries and execute web searches. "
                "Find the most relevant articles, lists, or directories for **both** categories. "
                "Prioritize sources listing multiple companies relevant to either HR globally or B2B within New England. "
                "Filter out purely national lists unless they specifically segment by NE or are highly relevant HR lists. "
                "Return a combined list of unique, relevant URLs."
            ),
            expected_output=(
                # UPDATED Expected Output (consider increasing limit in config)
                "A Python list of unique, relevant URLs (up to 10 or as configured), potentially covering both HR and NE B2B sources."
            ),
            agent=research_agent, # Use the validated single research agent
            context=[plan_search_task] # Context depends on plan_search_task being valid
        )
        logger.debug("Unified search tasks created successfully.")
        return [plan_search_task, execute_search_task]

    except Exception as e:
        logger.error(f"Error creating unified search tasks: {e}", exc_info=True)
        return [] # Return empty list on error


# --- EXTRACTION TASK (Unified for Research Agent) ---

def create_extraction_task(url, agents):
    """
    Create a task for the Research Agent to extract company information
    from a given URL (could be HR or NE B2B source).
    """
    logger.info(f"Creating extraction task for URL: {url}")
    extraction_task = None
    research_agent = agents.get('research') # Still uses the single Research Agent

    if not isinstance(research_agent, Agent):
         logger.error(f"Research agent not found or invalid in create_extraction_task for URL {url}. Cannot create task.")
         return None

    try:
        extraction_task = Task(
            description=(
                # Kept general, removed HR-specific examples
                f"Use the Generic Scraper tool to scrape the content from: {url}\n"
                "Analyze the text content to identify companies mentioned. "
                "These could be HR-related companies OR general B2B companies located in New England. "
                "For each company identified, determine their name and website URL. "
                "Return the results as a Python list of dictionaries in the format: "
                "[{'name': 'Company Name', 'website': 'https://company-website.com'}, ...]"
                # Note: Consider adding classification/tagging here in future if needed
            ),
            expected_output=(
                "A Python list of dictionaries containing company names and website URLs. "
                "The list should contain companies identified on the page, regardless of type (HR or B2B)."
            ),
            agent=research_agent # Use validated research agent
        )
        logger.debug(f"Extraction task created successfully for {url}.")
        return extraction_task
    except Exception as e:
        logger.error(f"Error creating extraction task for {url}: {e}", exc_info=True)
        return None


# --- ANALYSIS TASK (Requires specific HR or NE_B2B Analysis Agent) ---

def create_analysis_task(company_name, company_website, agent: Agent): # MODIFIED: Added agent parameter
    """
    Create a task to analyze a company (find email and identify challenges/opportunities)
    using the **provided** analysis agent (either HR or NE_B2B specific).
    """
    logger.info(f"Creating analysis task for company: {company_name} using agent: {agent.role}")
    analysis_task = None

    # --- Check for Agent Validity ---
    if not isinstance(agent, Agent):
        logger.error(f"Invalid agent object provided to create_analysis_task for {company_name}. Cannot create task.")
        return None # Return None if agent is invalid
    # --- End Check ---

    # Determine context based on the agent's role (this helps tailor the description)
    is_hr_context = "HR Company Analyzer" in agent.role
    is_ne_b2b_context = "New England B2B Company Analyzer" in agent.role

    # UPDATED Description: Tailored based on the agent being used
    task_description = (
        f"Analyze the company '{company_name}' with website '{company_website}'.\n"
        "1. Use the Unified Email Finder tool to find a contact email address for the company.\n"
        "2. Use the Company Pain Point Analyzer tool to identify 3-5 potential business challenges OR opportunities this company might face. "
    )

    if is_hr_context:
        task_description += (
            "**Context: This is an HR-focused company.** Consider aspects relevant to their specific HR niche "
            "(e.g., competition for consultants, changing regulations for compliance firms, technology adoption for software vendors). "
            "Think about why attending or sponsoring an HR conference could be **directly beneficial within their industry** "
            "(e.g., networking with peers/buyers, HR tech lead gen, brand visibility, learning industry trends)."
        )
    elif is_ne_b2b_context:
        task_description += (
            "**Context: This is likely a general B2B company in New England.** Consider why this *type* of B2B company "
            "might want to reach **HR professionals or business leaders** who attend regional HR conferences "
            "(e.g., selling their B2B services/products to HR departments or regional companies, recruiting talent "
            "showcased at HR events, building brand awareness among regional business leaders, networking with potential large clients "
            "whose HR teams attend). Frame the points with this specific HR conference sponsorship context in mind."
        )
    else:
         # Fallback for safety, although agent roles should match
         task_description += (
             "Consider potential business challenges or opportunities and why attending or sponsoring an HR conference could be beneficial."
         )

    task_description += "\nReturn the results in a structured format with the email and analysis points clearly labeled."


    try:
        analysis_task = Task(
            description=task_description, # Use the dynamically generated description
            expected_output=(
                # UPDATED Expected Output
                "A structured response containing the company's contact email and identified potential business "
                "challenges/opportunities relevant to HR conference sponsorship, **appropriately contextualized** "
                "based on the company type (HR-specific or general NE B2B)."
            ),
            agent=agent # Use the **passed-in** agent object (hr_analysis or ne_b2b_analysis)
        )
        logger.debug(f"Analysis task created successfully for {company_name} using agent {agent.role}: Type {type(analysis_task)}")
        return analysis_task
    except Exception as e:
        logger.error(f"Unexpected error creating analysis task for {company_name} using agent {agent.role}: {e}", exc_info=True)
        return None # Return None if Task() fails


# --- REVIEW TASK (Requires specific HR or NE_B2B Reviewer Agent) ---

def create_review_task(company_name, company_website, initial_pain_points, agent: Agent): # MODIFIED: Added agent parameter
    """
    Create a task to review and refine initially generated pain points
    using the **provided** reviewer agent (either HR or NE_B2B specific).
    """
    logger.info(f"Creating review task for company: {company_name} using agent: {agent.role}")
    review_task = None

    # --- Check for Agent Validity ---
    if not isinstance(agent, Agent):
        logger.error(f"Invalid agent object provided to create_review_task for {company_name}. Cannot create task.")
        return None # Return None if agent is invalid
    # --- End Check ---

    # Determine context based on the agent's role
    is_hr_context = "HR Lead Quality" in agent.role
    is_ne_b2b_context = "NE B2B Lead Quality" in agent.role

    formatted_initial_points = "\n".join([f"- {p.strip()}" for p in initial_pain_points.split('\n') if p.strip()])
    if not formatted_initial_points:
        formatted_initial_points = "No initial pain points provided."

    # UPDATED Description: Tailored based on the agent being used
    task_description = (
        f"Review the initially generated pain points for the company '{company_name}' (Website: {company_website}).\n\n"
        f"Initial Pain Points Provided:\n{formatted_initial_points}\n\n"
    )

    if is_hr_context:
         task_description += (
             "**Context: This is an HR-focused company.** "
             "1. Assess these points: Are they specific to a company in the HR sector? Are they generic business platitudes? Do they clearly link to the value of sponsoring an *HR industry* conference?\n"
             "2. If points are too generic (e.g., 'facing competition'), refine them into specific HR challenges/opportunities (e.g., 'Intensifying competition from specialized boutique HR consultancies' or 'Need to differentiate new payroll module').\n"
             "3. If refinement isn't possible or the initial points are good and relevant to HR conference sponsorship, explain why.\n"
             "4. Provide a final list of 3-5 refined or validated pain points clearly justifying HR conference sponsorship for *this type* of HR company."
         )
    elif is_ne_b2b_context:
         task_description += (
             "**Context: This is likely a general B2B company in New England.** "
             "1. Assess these points: Are they specific to the company? Crucially, do they present a *plausible and clear reason* why this *non-HR* company would benefit from sponsoring an *HR conference* in the region (e.g., reaching NE HR decision-makers, regional brand building, local talent acquisition)?\n"
             "2. If points are generic or the link to HR conference value is weak (e.g., 'need more customers'), refine them to create a specific angle connecting their B2B offering/needs to the HR conference audience (e.g., 'Opportunity to network with NE HR leaders responsible for purchasing [Their Service/Product]' or 'Challenge in recruiting skilled local [Job Role] which HR conference attendees might influence').\n"
             "3. If refinement isn't possible or initial points strongly justify sponsorship for a B2B company, explain why.\n"
             "4. Provide a final list of 3-5 refined or validated pain points clearly justifying HR conference sponsorship for *this specific B2B company*."
         )
    else:
        # Fallback for safety
        task_description += (
            "1. Assess these points for specificity and relevance.\n"
            "2. Refine generic points if possible.\n"
            "3. Provide a final list of 3-5 refined or validated pain points."
        )

    task_description += "\nClearly label the final list."

    try:
        review_task = Task(
            description=task_description, # Use the dynamically generated description
            expected_output=(
                 # UPDATED Expected Output
                "A final, refined list of 3-5 specific business pain points or opportunities relevant to the company, "
                "clearly labeled and **explicitly justifying HR conference sponsorship within the appropriate context** (HR industry or NE B2B). "
                "Include a brief justification if points were significantly changed or kept as is."
            ),
            agent=agent # Use the **passed-in** agent object (hr_reviewer or ne_b2b_reviewer)
        )
        logger.debug(f"Review task created successfully for {company_name} using agent {agent.role}: Type {type(review_task)}")
        return review_task
    except Exception as e:
        logger.error(f"Unexpected error creating review task for {company_name} using agent {agent.role}: {e}", exc_info=True)
        return None # Return None if Task() fails
