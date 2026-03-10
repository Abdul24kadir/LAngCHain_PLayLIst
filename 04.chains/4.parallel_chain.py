from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableParallel
# note : parallel chains now are created using simple dic instead of runnables
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")        

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=api_token,
    max_new_tokens=450,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template="write a detailed summary on the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="create 5 questions and answers on the {topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template="merge the following summary and quiz in a single document \n summary->{summary} \n quiz->{quiz}",
    input_variables=['summary','quiz']
)

parser = StrOutputParser()


# parallel_chain = RunnableParallel({
#     'summary':prompt1 | model | parser,
#     'quiz' : prompt2 | model | parser
# })

parallel_chain = {
    'summary':prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
}

merge_chain = prompt3 | model | parser 


chain = parallel_chain | merge_chain

topic = """
Cricket, often described as a game of glorious uncertainties, is far more than a sport; it is a cultural phenomenon, a colonial legacy transformed into a postcolonial passion, and a peculiar blend of athleticism, strategy, and almost theatrical drama that has captivated billions across the globe. Born in the rural pastures of 16th-century England as a children’s pastime involving a bat and a ball, it evolved by the 17th century into an organised adult pursuit patronised by aristocrats and gamblers alike. The first recorded laws were drawn up in 1744, and the Marylebone Cricket Club (MCC), founded in 1787, became the game’s de facto guardian, codifying rules that still form the backbone of the modern Laws of Cricket. From sleepy English village greens, the game travelled the British Empire’s trade routes and colonial corridors, taking deepest root wherever red dust and hot sun prevailed: Australia, South Africa, the West Indies, India, Pakistan, Sri Lanka, and later Bangladesh and Afghanistan. In these nations, cricket transcended mere recreation to become an instrument of identity, resistance, and nation-building. Nowhere is this more evident than in the Indian subcontinent, where the 1983 World Cup victory under Kapil Dev remains a foundational myth of modern Indian self-belief, and where the Indian Premier League (IPL), launched in 2008, turned the sport into a billion-dollar entertainment behemoth that redefined player salaries, franchise ownership, and the very tempo of the game.
At its heart, cricket is a contest between bat and ball played between two teams of eleven players each on an oval field whose centrepiece is a 22-yard (20.12-metre) rectangular pitch. The bowler delivers the hard, leather-covered ball (weighing about 160 grams) at speeds that can exceed 150 km/h, attempting to hit the wicket—three wooden stumps topped by two bails—guarded by the batsman armed only with a flat-faced willow bat. A batsman can be dismissed in ten different ways, from being bowled or caught to the infamous leg-before-wicket (lbw), run-out, or the rare “handled the ball” or “timed out.” The sheer variety of dismissals, combined with the fact that a single innings can last anything from twenty overs to five days, gives cricket its unique temporal elasticity. Test cricket, played over five days with no limit on overs per innings, is the purest and most demanding format, requiring endurance, concentration, and tactical nuance that few sports can match. One-day internationals (50 overs per side) and Twenty20 (20 overs) condense the drama into a few hours, prioritising aggression and entertainment; the shortest format has produced modern icons like Chris Gayle’s whirlwind centuries and Rashid Khan’s mesmerising leg-spin variations.
The game’s global map is strikingly asymmetrical. While England and Australia contested the first Test in 1877 and inaugurated the Ashes urn in 1882 (after a satirical obituary declared English cricket “dead” following an Australian victory), the balance of power has decisively shifted southward. Since the turn of the millennium, India’s economic might—fuelled by broadcasting rights and a population that treats cricketers as demigods—has made the Board of Control for Cricket in India (BCCI) the game’s financial superpower. The IPL’s 2023–2027 media rights cycle alone fetched over US$6 billion, dwarfing the revenues of most traditional sports leagues worldwide. Yet Test cricket, despite its declining commercial appeal in some quarters, retains a reverential status among purists. Epic rearguards like Steve Waugh’s Australians in the 1990s and 2000s, Brian Lara’s record-breaking 400 not out in 2004, or India’s triumphant chase of 378 in 2021 at the Gabba after being bowled out for 36 in the first innings remind us why the longest format is still called the ultimate test of character.
Cricket’s lexicon is a delightful curiosity in itself—silly mid-on, googly, chinaman, nightwatchman, reverse swing, doosra—terms that sound like entries from a Lewis Carroll poem yet describe precise technical realities. The game has produced literature of the highest order (Neville Cardus, C.L.R. James’s Beyond a Boundary, Mike Marqusee’s Anyone But England) and has been a stage for geopolitical theatre: the Bodyline series of 1932–33 that almost fractured Anglo-Australian relations, the rebel tours to apartheid South Africa, D’Oliveira Affair, and more recently the India–Pakistan encounters that stop two nuclear-armed nations in their tracks every time the teams meet. Today, as climate change shortens attention spans and franchise leagues proliferate (IPL, Big Bash, Hundred, SA20, ILT20, MLC), the challenge for cricket is to preserve its multifaceted soul—part ballet, part chess, part gladiatorial combat—while embracing the brash, floodlit, music-blasting energy of its shortest format. In an era of instant gratification, cricket remains that rare spectacle capable of unfolding slowly over five rain-interrupted days yet still leaving its devotees breathless when a single ball, like the one Shane Warne bowled to Mike Gatting in 1993—the “Ball of the Century” that drifted and spun impossibly—can redefine what is possible in sport.
"""
result = chain.invoke({'topic':topic})

print(result)

