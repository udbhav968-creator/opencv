"""
ai_verdict.py
-------------
AI-powered natural language verdict explainer for DRS decisions.
Generates a human-readable analysis of the ball trajectory and DRS outcome.
"""

def generate_verdict_explanation(pitching_zone, impact_zone, wicket_verdict, final_call):
    """
    Generate a natural language explanation of the DRS decision.
    
    Args:
        pitching_zone: Zone where ball pitched (e.g. 'OUTSIDE_LEG', 'IN_LINE', 'OUTSIDE_OFF')
        impact_zone:   Zone of pad impact
        wicket_verdict: Wicket prediction ('HITTING', 'UMPIRES_CALL', 'MISSING')
        final_call:    Final DRS decision ('OUT', 'NOT OUT', "UMPIRE'S CALL")
    
    Returns:
        dict with 'summary', 'reasoning', 'confidence', 'tips'
    """

    zone_map = {
        "OUTSIDE_LEG": "outside leg stump",
        "IN_LINE":      "in line with the stumps",
        "OUTSIDE_OFF":  "outside off stump",
    }
    wicket_map = {
        "HITTING":      "hitting the stumps",
        "UMPIRES_CALL": "clipping the edge of the stumps (Umpire's Call)",
        "MISSING":      "missing the stumps",
    }

    pitch_desc   = zone_map.get(pitching_zone,  pitching_zone)
    impact_desc  = zone_map.get(impact_zone,    impact_zone)
    wicket_desc  = wicket_map.get(wicket_verdict, wicket_verdict)

    # --- Confidence scoring ---
    confidence = _compute_confidence(pitching_zone, impact_zone, wicket_verdict)

    # --- Reasoning ---
    reasoning_parts = [
        f"The ball pitched **{pitch_desc}** and struck the pad **{impact_desc}**.",
        f"Trajectory projection shows the ball would have gone on to be **{wicket_desc}**.",
    ]

    # LBW law notes
    if pitching_zone == "OUTSIDE_LEG":
        reasoning_parts.append(
            "Under LBW law, a ball pitching outside leg stump cannot be given out, "
            "regardless of impact or wicket projection."
        )
    elif pitching_zone == "OUTSIDE_OFF" and impact_zone != "IN_LINE":
        reasoning_parts.append(
            "The ball pitched and impacted outside off stump — "
            "no LBW dismissal is possible when the impact is also outside off."
        )
    elif wicket_verdict == "MISSING":
        reasoning_parts.append(
            "Since the ball is projected to miss the stumps entirely, "
            "no LBW dismissal can stand."
        )
    elif wicket_verdict == "UMPIRES_CALL":
        reasoning_parts.append(
            "The ball is only clipping the stumps — Umpire's Call applies "
            "and the on-field decision stands."
        )

    reasoning = " ".join(reasoning_parts)

    # --- Summary sentence ---
    emoji_map = {"OUT": "🔴", "NOT OUT": "🟢", "UMPIRE'S CALL": "🟡"}
    emoji = emoji_map.get(final_call, "⚪")
    summary = f"{emoji} **{final_call}** — {_build_summary(pitching_zone, impact_zone, wicket_verdict, final_call)}"

    # --- Tips for real footage ---
    tips = _get_tips(pitching_zone, impact_zone, wicket_verdict)

    return {
        "summary":    summary,
        "reasoning":  reasoning,
        "confidence": confidence,
        "tips":       tips,
        "final_call": final_call,
    }


def _compute_confidence(pitching_zone, impact_zone, wicket_verdict):
    """Return a confidence percentage based on how clear-cut the decision is."""
    if wicket_verdict == "UMPIRES_CALL":
        return 55  # borderline
    if pitching_zone == "OUTSIDE_LEG":
        return 98  # clear not-out
    if wicket_verdict == "MISSING":
        if impact_zone == "OUTSIDE_OFF":
            return 97
        return 88
    if wicket_verdict == "HITTING" and impact_zone == "IN_LINE" and pitching_zone == "IN_LINE":
        return 95
    if wicket_verdict == "HITTING" and pitching_zone == "IN_LINE":
        return 88
    return 75


def _build_summary(pitching_zone, impact_zone, wicket_verdict, final_call):
    if final_call == "NOT OUT":
        if pitching_zone == "OUTSIDE_LEG":
            return "ball pitched outside leg stump — batsman protected by law."
        if wicket_verdict == "MISSING":
            return "ball tracking shows it would have missed the stumps."
        return "insufficient evidence to overturn the on-field decision."
    if final_call == "OUT":
        return "ball tracking confirms the ball would have hit the stumps."
    return "ball is only clipping the stumps — on-field decision stands."


def _get_tips(pitching_zone, impact_zone, wicket_verdict):
    tips = []
    if pitching_zone == "OUTSIDE_LEG":
        tips.append("Always review pitching zone first — outside leg is an automatic NOT OUT.")
    if wicket_verdict == "UMPIRES_CALL":
        tips.append("Umpire's Call means the review is retained by the batting team.")
    if impact_zone == "OUTSIDE_OFF":
        tips.append("Impact outside off stump with no prior pitch in-line is also NOT OUT.")
    if not tips:
        tips.append("Check both the pitching zone and impact zone before reviewing a decision.")
    return tips
