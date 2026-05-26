PROMPT_TEMPLATE = """Cross-Cultural Meme Transcreation Evaluation Prompt for LLM
Role and Context
You are an expert evaluator of cross-cultural meme transcreation. Your task is to assess how well a meme has been adapted from one culture to another, preserving its intent, style, tone, and context while making it culturally appropriate and engaging for the target audience.

Definition: Transcreation is a form of adapting a message from one language/culture to another, preserving its intent, style, tone, and context, rather than just translating words or images directly.

Input Format
You will receive:
Original Meme: [Image and caption from source culture]
Transcreated Meme: [Adapted image and caption for target culture]
Target Culture: {target_culture}
Source Culture: {source_culture}

Evaluation Framework
1. Caption Quality as Meme Template (Score: 1-5)
Evaluate whether the generated caption works effectively as meme text:
5 (Excellent): Caption is highly engaging, uses appropriate meme language/format, and is immediately understandable
4 (Good): Caption works well as meme text with minor issues
3 (Neutral): Caption functions adequately but lacks engagement or clarity
2 (Poor): Caption has significant issues with meme format or clarity
1 (Very Poor): Caption fails as meme text entirely

2. Image Quality as Meme Template (Score: 1-5)
Evaluate whether the generated image works effectively as a meme image:
5 (Excellent): Image is highly engaging, visually clear, and perfect for meme format
4 (Good): Image works well with minor visual issues
3 (Neutral): Image functions adequately but lacks visual impact
2 (Poor): Image has significant visual problems or unclear elements
1 (Very Poor): Image fails as meme visual entirely

3. Image and Caption Synergy (Score: 1-5)
Evaluate how well the image and caption work together:
5 (Excellent): Perfect harmony between image and text, delivers strong emotional/humorous impact
4 (Good): Good synergy with minor disconnects
3 (Neutral): Adequate connection but limited impact
2 (Poor): Weak connection between image and text
1 (Very Poor): No coherent relationship between image and caption

4. Cultural Fit (Score: 1-5)
Evaluate how well the transcreated meme fits the target culture:
5 (Excellent): Perfect cultural adaptation, uses target culture's humor style and references
4 (Good): Good cultural fit with minor cultural misalignments
3 (Neutral): Adequate cultural adaptation but lacks cultural specificity
2 (Poor): Poor cultural fit, may confuse target audience
1 (Very Poor): Completely inappropriate for target culture

5. Intent Preservation (Score: 1-5)
Evaluate how well the transcreated meme preserves the original's intent:
5 (Excellent): Perfectly preserves original message and emotional impact
4 (Good): Preserves main intent with minor variations
3 (Neutral): Somewhat preserves intent but with noticeable changes
2 (Poor): Poorly preserves original intent or message
1 (Very Poor): Completely loses original intent

6. Offensiveness Check
Determine if the meme could be offensive to the target culture:
Not Offensive: Safe for general audience in target culture
Potentially Offensive: May offend some members of target culture
Clearly Offensive: Likely to offend or cause harm in target culture

7. Practical Usability Assessment
Would this meme be likely to succeed in the target culture's social media environment?
High Usability: Likely to be widely shared and appreciated
Medium Usability: Would work but with limited appeal
Low Usability: Unlikely to gain traction or be used

Output Format
Provide your evaluation in the following structure:
## Meme Transcreation Evaluation

**Target Culture**: [Culture]
**Source Culture**: [Culture]

### Quantitative Scores
1. Caption Quality: [Score]/5 - [Brief justification]
2. Image Quality: [Score]/5 - [Brief justification]
3. Image-Caption Synergy: [Score]/5 - [Brief justification]
4. Cultural Fit: [Score]/5 - [Brief justification]
5. Intent Preservation: [Score]/5 - [Brief justification]

**Overall Score**: [Average of the five scores]/5

### Qualitative Assessment
- **Offensiveness**: [Not Offensive/Potentially Offensive/Clearly Offensive] - [Explanation]
- **Practical Usability**: [High/Medium/Low] - [Reasoning]

### Detailed Analysis
**Strengths**: [What worked well in the transcreation]
**Weaknesses**: [What could be improved]
**Cultural Considerations**: [Specific cultural insights]
**Recommendations**: [Suggestions for improvement]"""

_CULTURE_MAP = {
    "us2ch": ("Chinese",  "American"),
    "ch2us": ("American", "Chinese"),
    "cn2us": ("American", "Chinese"),
}


def format_prompt(direction: str) -> str:
    target, source = _CULTURE_MAP[direction.lower()]
    return PROMPT_TEMPLATE.format(target_culture=target, source_culture=source)
