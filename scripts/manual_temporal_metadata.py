"""
Manually curated temporal metadata for all documents.
Based on content analysis of each file.
"""

from datetime import datetime

# Current ingestion timestamp (will be set when script runs)
INGESTION_DATE = datetime.now().isoformat()

# Manual metadata mapping based on document content analysis
# All dates in ISO 8601 format
TEMPORAL_METADATA = {
    "2care_implementation_methodology.txt": {
        "modified_date": "2026-01-20T13:06:39.960908",  # From filesystem
        "ingestion_date": INGESTION_DATE,  # Will be set at runtime
        "content_date": "2026-01-01T00:00:00"  # From "Last Updated: January 2026" - using first day of month
    },
    "2care_q4_2024_financial_report.txt": {
        "modified_date": "2026-01-20T12:49:05.118817",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2024-10-01T00:00:00"  # Q4 2024 starts Oct 1, 2024 (Reporting Period: Oct 1 - Dec 31, 2024)
    },
    "2care_api_technical_documentation.txt": {
        "modified_date": "2026-01-20T13:07:40.366573",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2026-01-01T00:00:00"  # From "Last Updated: January 2026"
    },
    "2care_company_overview.txt": {
        "modified_date": "2026-01-20T12:48:53.692817",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2025-12-31T00:00:00"  # No explicit date, but mentions current stats - likely end of 2025
    },
    "2care_customer_success_stories.txt": {
        "modified_date": "2026-01-20T12:55:27.251333",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2026-01-01T00:00:00"  # From "Last Updated: January 2026"
    },
    "2care_engineering_team_structure.txt": {
        "modified_date": "2026-01-20T12:51:10.372351",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2025-01-01T00:00:00"  # From "Last Updated: January 2025"
    },
    "2care_product_documentation_ehr.txt": {
        "modified_date": "2026-01-20T12:49:15.612937",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2024-12-01T00:00:00"  # From "Last Updated: December 2024"
    },
    "2care_sales_playbook.txt": {
        "modified_date": "2026-01-20T12:50:33.086832",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2025-01-01T00:00:00"  # From "Effective Date: January 2025"
    },
    "2care_security_compliance_guide.txt": {
        "modified_date": "2026-01-20T12:55:54.283817",
        "ingestion_date": INGESTION_DATE,
        "content_date": "2026-01-01T00:00:00"  # From "Last Updated: January 2026"
    }
}

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Update ingestion_date with current timestamp
    current_ingestion = datetime.now().isoformat()
    for file_meta in TEMPORAL_METADATA.values():
        file_meta["ingestion_date"] = current_ingestion
    
    # Print formatted output
    print("=" * 80)
    print("TEMPORAL METADATA FOR ALL DOCUMENTS")
    print("=" * 80)
    print(f"\nIngestion Date: {current_ingestion}\n")
    
    for filename, metadata in sorted(TEMPORAL_METADATA.items()):
        print(f"📄 {filename}")
        print(f"   Modified Date:  {metadata['modified_date']}")
        print(f"   Ingestion Date: {metadata['ingestion_date']}")
        print(f"   Content Date:   {metadata['content_date']}")
        print()
    
    print("=" * 80)
    print("\n📋 JSON Output:\n")
    print(json.dumps(TEMPORAL_METADATA, indent=2))
    
    # Save to file
    output_file = Path(__file__).parent.parent / "data" / "temporal_metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(TEMPORAL_METADATA, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")

