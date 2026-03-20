"""
Preference items for persona probing.

60 everyday preference items across 10 domains, each expressed as a binary like/dislike.
These are designed to be concrete, relatable, and span diverse aspects of personality.
"""

PREFERENCE_ITEMS = {
    # Food & Drink (items 0-5)
    0: {"domain": "food", "like": "loves spicy food", "dislike": "dislikes spicy food"},
    1: {"domain": "food", "like": "prefers home-cooked meals", "dislike": "prefers eating out at restaurants"},
    2: {"domain": "food", "like": "enjoys trying exotic cuisines", "dislike": "prefers familiar, traditional foods"},
    3: {"domain": "food", "like": "loves coffee", "dislike": "prefers tea over coffee"},
    4: {"domain": "food", "like": "enjoys fine dining", "dislike": "prefers casual, simple meals"},
    5: {"domain": "food", "like": "is vegetarian/plant-based", "dislike": "enjoys meat and barbecue"},

    # Music & Arts (items 6-11)
    6: {"domain": "arts", "like": "loves classical music", "dislike": "finds classical music boring"},
    7: {"domain": "arts", "like": "enjoys abstract art", "dislike": "prefers realistic/representational art"},
    8: {"domain": "arts", "like": "loves live concerts and festivals", "dislike": "prefers listening to music alone at home"},
    9: {"domain": "arts", "like": "plays a musical instrument", "dislike": "has no interest in playing instruments"},
    10: {"domain": "arts", "like": "enjoys poetry and literary fiction", "dislike": "prefers non-fiction and practical reading"},
    11: {"domain": "arts", "like": "loves indie/alternative music", "dislike": "prefers mainstream pop music"},

    # Social & Interpersonal (items 12-17)
    12: {"domain": "social", "like": "loves large social gatherings and parties", "dislike": "prefers small, intimate gatherings"},
    13: {"domain": "social", "like": "enjoys meeting new people", "dislike": "prefers spending time with close friends only"},
    14: {"domain": "social", "like": "loves hosting dinner parties", "dislike": "prefers being a guest rather than hosting"},
    15: {"domain": "social", "like": "enjoys team sports", "dislike": "prefers solo activities"},
    16: {"domain": "social", "like": "is very active on social media", "dislike": "avoids social media"},
    17: {"domain": "social", "like": "loves public speaking", "dislike": "dreads public speaking"},

    # Outdoors & Nature (items 18-23)
    18: {"domain": "outdoors", "like": "loves hiking and camping", "dislike": "prefers indoor activities"},
    19: {"domain": "outdoors", "like": "enjoys gardening", "dislike": "has no interest in gardening"},
    20: {"domain": "outdoors", "like": "loves the beach and ocean", "dislike": "prefers mountains over beaches"},
    21: {"domain": "outdoors", "like": "enjoys extreme sports (skydiving, rock climbing)", "dislike": "prefers safe, low-risk activities"},
    22: {"domain": "outdoors", "like": "loves animals and pets", "dislike": "is indifferent to animals"},
    23: {"domain": "outdoors", "like": "prefers warm, sunny weather", "dislike": "prefers cool, rainy weather"},

    # Intellectual & Learning (items 24-29)
    24: {"domain": "intellectual", "like": "loves science and technology", "dislike": "prefers humanities and liberal arts"},
    25: {"domain": "intellectual", "like": "enjoys philosophical debates", "dislike": "finds philosophical discussions tedious"},
    26: {"domain": "intellectual", "like": "loves puzzles and brain teasers", "dislike": "finds puzzles frustrating"},
    27: {"domain": "intellectual", "like": "enjoys learning new languages", "dislike": "has no interest in language learning"},
    28: {"domain": "intellectual", "like": "loves documentaries", "dislike": "prefers fictional entertainment"},
    29: {"domain": "intellectual", "like": "enjoys data and statistics", "dislike": "prefers intuition over data"},

    # Values & Lifestyle (items 30-35)
    30: {"domain": "values", "like": "values routine and predictability", "dislike": "craves spontaneity and surprise"},
    31: {"domain": "values", "like": "is very environmentally conscious", "dislike": "prioritizes convenience over environmental impact"},
    32: {"domain": "values", "like": "values tradition and heritage", "dislike": "embraces change and innovation"},
    33: {"domain": "values", "like": "is highly competitive", "dislike": "prefers cooperation over competition"},
    34: {"domain": "values", "like": "is a morning person", "dislike": "is a night owl"},
    35: {"domain": "values", "like": "values minimalism and simplicity", "dislike": "enjoys collecting things and abundance"},

    # Entertainment (items 36-41)
    36: {"domain": "entertainment", "like": "loves horror movies", "dislike": "avoids horror movies"},
    37: {"domain": "entertainment", "like": "enjoys video games", "dislike": "has no interest in video games"},
    38: {"domain": "entertainment", "like": "loves stand-up comedy", "dislike": "prefers dramatic performances"},
    39: {"domain": "entertainment", "like": "enjoys reality TV", "dislike": "dislikes reality TV"},
    40: {"domain": "entertainment", "like": "loves board games and card games", "dislike": "finds board games tedious"},
    41: {"domain": "entertainment", "like": "enjoys anime and manga", "dislike": "has no interest in anime"},

    # Travel & Exploration (items 42-47)
    42: {"domain": "travel", "like": "loves international travel", "dislike": "prefers staying close to home"},
    43: {"domain": "travel", "like": "prefers luxury travel", "dislike": "prefers budget/backpacking travel"},
    44: {"domain": "travel", "like": "enjoys road trips", "dislike": "prefers flying to destinations"},
    45: {"domain": "travel", "like": "loves exploring cities", "dislike": "prefers rural/nature destinations"},
    46: {"domain": "travel", "like": "enjoys solo travel", "dislike": "prefers traveling with companions"},
    47: {"domain": "travel", "like": "loves trying local street food while traveling", "dislike": "sticks to familiar foods while traveling"},

    # Work & Productivity (items 48-53)
    48: {"domain": "work", "like": "thrives in fast-paced environments", "dislike": "prefers calm, steady work pace"},
    49: {"domain": "work", "like": "loves working in teams", "dislike": "prefers working independently"},
    50: {"domain": "work", "like": "enjoys leadership roles", "dislike": "prefers individual contributor roles"},
    51: {"domain": "work", "like": "values work-life balance above career advancement", "dislike": "is highly career-driven and ambitious"},
    52: {"domain": "work", "like": "loves creative/artistic work", "dislike": "prefers analytical/systematic work"},
    53: {"domain": "work", "like": "enjoys multitasking", "dislike": "prefers deep focus on one task"},

    # Technology & Digital (items 54-59)
    54: {"domain": "tech", "like": "is an early adopter of new technology", "dislike": "prefers proven, established technology"},
    55: {"domain": "tech", "like": "loves coding and programming", "dislike": "has no interest in coding"},
    56: {"domain": "tech", "like": "enjoys virtual reality experiences", "dislike": "prefers real-world experiences"},
    57: {"domain": "tech", "like": "loves sci-fi and futurism", "dislike": "prefers historical settings and nostalgia"},
    58: {"domain": "tech", "like": "is privacy-conscious online", "dislike": "freely shares personal information online"},
    59: {"domain": "tech", "like": "loves smart home gadgets", "dislike": "prefers simple, non-connected devices"},
}

# Follow-up probe questions designed to test consistency
PROBE_QUESTIONS = [
    "What would be your ideal way to spend a free Saturday?",
    "If you could plan your dream vacation, what would it look like?",
    "What kind of gift would make you happiest?",
    "Describe your ideal living environment.",
    "What would you choose for a perfect evening of entertainment?",
]

NUM_ITEMS = len(PREFERENCE_ITEMS)
assert NUM_ITEMS == 60
