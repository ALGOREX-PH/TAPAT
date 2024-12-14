from swarm import Swarm

from agents.PersonalizationAgent import PersonalizationAgent
from agents.AssessmentAgent import AssessmentAgent

client = Swarm()
personalization_agent = PersonalizationAgent()
assessment_agent = AssessmentAgent()

linkedin_data = {
    'years_experience': 3.5,
    'programming_languages': ['python', 'javascript'],
    'job_titles': ['Software Engineer'],
    'projects': ['n/a'],
    'current_role': 'AI Software Engineer'
}

github_assessment = "Regular GitHub activity with JavaScript/Node.js focus. 3 full-stack projects using React and Express. Clean code organization and documentation. Averages 3 commits/week. Strong in web development with demonstrated interest in data visualization libraries."

# personalization_response = client.run(
#     agent=personalization_agent,
#     messages=[
#         {"role": "user", "content": f"Generate marketing strategies for the following student profile {linkedin_data} and assessment {github_assessment}."}
#     ],
#     context_variables={"profile_data": linkedin_data,
#                         "assessment": github_assessment}
# )


# personalization_strategies = personalization_response.messages[-1]["content"]
# print(personalization_strategies)
user_input = f"""Please assess this candidate based on the following information:

LinkedIn Assessment:
- Years of Experience: {linkedin_data['years_experience']}
- Programming Languages: {', '.join(linkedin_data['programming_languages'])}
- Job Titles: {', '.join(linkedin_data['job_titles'])}
- Projects: {', '.join(linkedin_data['projects'])}
- Current Role: {linkedin_data['current_role']}

GitHub Assessment:
{github_assessment}

Provide your assessment in exactly this format:
CANDIDATE ASSESSMENT SUMMARY
--------------------------
Eligibility: [Recommended/Not Recommended]
Target Category: [Scholarship/Paid]
Reasoning: [Your comprehensive reasoning combining LinkedIn and GitHub insights]"""

assessment_response = client.run(
    agent=assessment_agent,
    messages=[{"role": "user", "content": user_input}],
    context_variables={"linkedin_data": linkedin_data,
                        "github_assessment": github_assessment}
    
)

student_assessment = assessment_response.messages[-1]["content"]
print(student_assessment)
