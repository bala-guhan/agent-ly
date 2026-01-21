def get_date_extraction_prompt(query: str, conversation_context: str = "") -> str:
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    context_section = ""
    if conversation_context:
        context_section = f"\n\nConversation Context:\n{conversation_context}\n\nUse this context to understand references like 'the quarter', 'that period', 'it', etc."
    
    return f"""Extract date range from the query ONLY if temporal filtering is needed (query mentions specific time period).

Current date context: Today is in {current_year}, month {current_month}.

Query: "{query}"{context_section}

Only extract dates if query mentions or context indicates:
- Years: "2024", "2025", "in 2024"
- Quarters: "Q4 2024", "Q1 2025", "Q4" (without year defaults to most recent Q4 = Q4 {current_year})
- Months: "January 2024", "March 2025"
- Date ranges: "between X and Y", "from X to Y"
- Time periods: "last 6 months", "this year"
- References: "the quarter", "that period" (if context clarifies)

IMPORTANT: If a quarter is mentioned without a year (e.g., "Q4", "Q1"), default to the most recent occurrence:
- "Q4" without year → Q4 {current_year} (date_start="{current_year}-10-01", date_end="{current_year}-12-31")
- "Q1" without year → Q1 {current_year} (date_start="{current_year}-01-01", date_end="{current_year}-03-31")
- "Q2" without year → Q2 {current_year} (date_start="{current_year}-04-01", date_end="{current_year}-06-30")
- "Q3" without year → Q3 {current_year} (date_start="{current_year}-07-01", date_end="{current_year}-09-30")

If query/context mentions dates/time periods, convert to ISO format (YYYY-MM-DD):
- Year: "2024" → date_start="2024-01-01", date_end="2024-12-31"
- Quarter: "Q4 2024" → date_start="2024-10-01", date_end="2024-12-31"
- Quarter without year: "Q4" → date_start="{current_year}-10-01", date_end="{current_year}-12-31"
- Month: "January 2024" → date_start="2024-01-01", date_end="2024-01-31"
- Range: "between 2024-06-01 and 2024-12-31" → extract both dates

If no dates/time periods mentioned or inferred from context, return date_start=None, date_end=None

Extract dates:"""

