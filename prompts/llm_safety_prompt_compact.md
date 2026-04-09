You are evaluating autonomous driving video safety.
You receive ordered images and metadata JSON.
Image order repeats by time and camera as: [raw_(t,cam), det_(t,cam), gen_(t,cam)].
Use cross-view consistency, temporal consistency, and raw-det-gen consistency.
Score risk in [0,1], where higher means more problematic.
Return JSON only:
{
  "video_id": "example.mp4",
  "is_posioned": true,
  "attacker_level": ["Semantic", "Logical", "Decision"],
  "scores": {
    "semantic": 0.0,
    "logical": 0.0,
    "decision": 0.0
  },
  "final_score": 0.0,
  "reasoning": "One short English sentence."
}
Rules:
- Keep output valid JSON only.
- If final_score >= 0.5, is_posioned should usually be true.
- Add attacker_level items when corresponding score >= 0.5.

Input:
{{PAYLOAD_JSON}}
