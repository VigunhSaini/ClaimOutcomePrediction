"""
bonus_appeal.py
Retrieves appeal guidance snippets for common insurance claim denial reasons.
Includes keyword-based matching and an explicit 'matched: False' fallback for LLM safety.
"""

import re
from typing import Any, Dict


# Pre-defined policy guidance snippets for common healthcare denial codes
APPEAL_KNOWLEDGE_BASE = {
    "missing_auth": {
        "keywords": ["auth", "authorization", "prior_auth", "precertification", "approval"],
        "guidance": (
            "Review pre-authorization log. If emergency care was rendered, cite retroactive "
            "authorization exceptions under ERISA/ACA guidelines. Attach physician clinical notes "
            "and hospital admission summary demonstrating medical urgency."
        ),
    },
    "eligibility_lapse": {
        "keywords": ["eligibility", "lapse", "terminated", "inactive", "coverage", "policy_term"],
        "guidance": (
            "Verify patient coverage active date on date of service. Obtain updated primary/secondary "
            "COB (Coordination of Benefits) documentation and submit state insurance verification snapshot."
        ),
    },
    "duplicate_claim": {
        "keywords": ["duplicate", "repeated", "exact", "already_processed", "same_service"],
        "guidance": (
            "Differentiate distinct service sessions using appropriate modifiers (e.g., -76 repeat procedure, "
            "-59 distinct procedural service). Provide itemized timestamped surgical logs or treatment sheets."
        ),
    },
    "timely_filing": {
        "keywords": ["timely", "filing_limit", "deadline", "late_submission", "expired"],
        "guidance": (
            "Provide electronic clearinghouse 999/277 acceptance reports proving initial transmission within "
            "payer's contractual timely filing window (e.g., 90 or 180 days). Cite payer manual clause."
        ),
    },
    "medical_necessity": {
        "keywords": ["necessity", "not_medically_necessary", "experimental", "lcd", "ncd", "clinical_criteria"],
        "guidance": (
            "Submit treating provider's Letter of Medical Necessity alongside relevant LCD/NCD coverage articles, "
            "radiology reports, conservative treatment failure history, and peer-reviewed clinical documentation."
        ),
    },
    "coding_mismatch": {
        "keywords": ["coding", "mismatch", "unbundling", "cci", "modifier", "diagnosis_inconsistent"],
        "guidance": (
            "Verify ICD-10 to CPT linkage. Cross-reference NCCI edits to ensure mutually exclusive code pairs "
            "are not unbundled. Include operative notes detailing independent anatomical sites."
        ),
    },
    "out_of_network": {
        "keywords": ["oon", "out_of_network", "network", "non_par", "no_surprises"],
        "guidance": (
            "Invoke No Surprises Act protections for emergency services or non-participating providers at in-network "
            "facilities. Request QPA (Qualifying Payment Amount) recalculation and independent dispute resolution (IDR)."
        ),
    },
    "bundled_service": {
        "keywords": ["bundle", "bundled", "inclusive", "global_period", "post_op"],
        "guidance": (
            "Check if service occurred outside standard global surgery period. If unrelated evaluation occurred during "
            "global window, append modifier -24 or -25 supported by separate clinical encounter documentation."
        ),
    },
}

DEFAULT_FALLBACK_GUIDANCE = (
    "Standard appeal procedure: Request formal written explanation of benefits (EOB) clarification, "
    "gather complete certified patient medical charts, and file a standard first-level administrative appeal "
    "within payer deadline."
)


def get_appeal_guidance(denial_reason: str) -> Dict[str, Any]:
    """
    Matches a raw denial reason string against known denial categories.
    
    Returns a dictionary with:
      - 'matched': bool (False flags fallback, critical for prompt grounding)
      - 'category': matched category key or None
      - 'guidance': specific appeal action snippet
    """
    if not isinstance(denial_reason, str):
        return {
            "matched": False,
            "category": None,
            "guidance": DEFAULT_FALLBACK_GUIDANCE,
        }

    # Normalize input: lowercase, replace punctuation/underscores with space
    normalized_input = re.sub(r"[_\W]+", " ", denial_reason.lower()).strip()
    words = set(normalized_input.split())

    best_match = None
    max_keyword_matches = 0

    for category, entry in APPEAL_KNOWLEDGE_BASE.items():
        match_count = sum(1 for kw in entry["keywords"] if kw in normalized_input or kw in words)
        if match_count > max_keyword_matches:
            max_keyword_matches = match_count
            best_match = category

    if best_match and max_keyword_matches > 0:
        return {
            "matched": True,
            "category": best_match,
            "guidance": APPEAL_KNOWLEDGE_BASE[best_match]["guidance"],
        }

    return {
        "matched": False,
        "category": None,
        "guidance": DEFAULT_FALLBACK_GUIDANCE,
    }


if __name__ == "__main__":
    test_queries = [
        "Prior authorization was missing for this procedure",
        "duplicate_claim_submitted_twice",
        "service considered experimental and not medically necessary",
        "unrecognized_random_rejection_code_xyz",
    ]

    print("Running appeal guidance snippet matching tests:\n")
    for q in test_queries:
        res = get_appeal_guidance(q)
        print(f"Query: '{q}'\n  -> Matched: {res['matched']} | Category: {res['category']}\n  -> Guidance: {res['guidance'][:80]}...\n")

