# main_script.py

from julep import Client
from dotenv import load_dotenv
import os
import textwrap
from character_info import characters_info


def initialize_julep_client():
    load_dotenv()
    base_url = os.getenv("JULEP_API_URL")
    api_key = os.getenv("JULEP_API_KEY")
    client = Client(api_key=api_key, base_url=base_url)
    return client


def get_character_reply(character_name, user_msg):
    client = initialize_julep_client()

    character_info = characters_info.get(character_name)
    if not character_info:
        return "Character not found."

    name = character_name
    agent_about = character_info["about"]
    situation_prompt = character_info["situation_prompt"]

    default_settings = {
        "temperature": 0.7,
        "top_p": 1,
        "min_p": 0.01,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "length_penalty": 1.0,
        "max_tokens": 150,
    }

    agent = client.agents.create(
        name=name,
        about=agent_about,
        default_settings=default_settings,
        model="gpt-4",
        tools=[]
    )

    about_user = """Average nerdy techbro/girl who spends 8 hours a day in front of a laptop.
    Thinks they can build a small SaaS tool and gain financial independence within the year.
    """
    user = client.users.create(
        name="Me",
        about=about_user,
    )

    session = client.sessions.create(
        user_id=user.id, agent_id=agent.id, situation=situation_prompt
    )

    response = client.sessions.chat(
        session_id=session.id,
        messages=[
            {
                "role": "user",
                "content": user_msg,
                "name": "Anon",
            }
        ],
        recall=True,
        remember=True,
    )

    return "\n".join(textwrap.wrap(response.response[0][0].content, width=100))


# Example usage:
user_msg = "hii Gojo will you marry me?? "
# You can change this to any character name you want
character_name = "Gojo"
reply = get_character_reply(character_name, user_msg)
print(reply)
