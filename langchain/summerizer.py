############## PROBLEM STATEMENT
### Summerize a long Youtube Videos
############## PROBLEM SOLVING APPROACH
### Step1: 

import os
import json
import evaluate
import re
import tiktoken
import textwrap
import yt_dlp
import whisper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import YouTubeVideo
from pytubefix import YouTube

from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from utility.llm import LLM

class YouTubeTranscripter:
    destination = '/mnt/c/Users/bibhu/OneDrive/Workspace/devops/genai/_data/'

    @staticmethod
    def __download_video(url):
        # destination path
        audio_path= YouTubeTranscripter.destination + "temp.mp3"

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(YouTubeTranscripter.destination, 'temp.%(ext)s'),
            'quiet': True
        }

        # Download the audio
        if os.path.exists(audio_path):
            print(f"{audio_path} exists. so no more download.")
        else :    
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info['title']

                # Get the actual file path
                new_file = ydl.prepare_filename(info)
                # Since we're converting to mp3, replace the extension
                new_file = new_file.rsplit(".", 1)[0] + ".mp3"
                #new_file = "temp.mp3"

                print(f"{title} has been successfully downloaded.")
                print(f"File saved as: {new_file}")

        return audio_path

    @staticmethod
    def __generate_scripts_from_audio(audio_path):
        script_file = YouTubeTranscripter.destination + "temp.txt"
        transcript = ""

        if os.path.exists(script_file):
            print(f"{script_file} exists. so no script generation again.")
            with open(script_file, "r") as file:
                transcript = file.read()
        else :
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            transcript = result["text"]

            script_file = YouTubeTranscripter.destination + "temp.txt"
            with open(script_file, "w") as file:
                file.write(transcript)

        return transcript
    
    @staticmethod
    def rate_the_summary(summary_response, transcripts):
        rater_system_message = """
            You are tasked with rating AI-generated summaries of youtube transcripts based on the given metric.
            You will be presented a transcript and an AI generated summary of the transcript as the input.
            In the input, the transcript will begin with ###transcript while the AI generated summary will begin with ###summary.

            Metric
            Check if the summary  is true to the transcript.
            The summary should cover all the aspects that are majorly being discussed in the transcript.
            The summary should be concise.

            Evaluation criteria:
            The task is to judge the extent to which the metric is followed by the summary.
            1 - The metric is not followed at all
            2 - The metric is followed only to a limited extent
            3 - The metric is followed to a good extent
            4 - The metric is followed mostly
            5 - The metric is followed completely

            Respond only with a rating between 0 and 5. Do not explain the rating.
        """

        rater_user_message_template = """
            ###transcript
            {transcript}

            ###summary
            {summary}
        """

        from utility.llm import LLM
        client = LLM.get_client()
        
        from langchain_core.output_parsers import StrOutputParser
        parser = StrOutputParser()

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate([
            ("system", rater_system_message),
            ("human", rater_user_message_template.format(
                    transcript=transcripts,
                    summary=summary_response
                )),
        ])

        chain = prompt | client | parser

        response = chain.invoke({"transcript": transcripts, "summary": summary_response},
                config={"temperature": 0, "max_tokens": 5, "model_name": "gpt-4o"})
        
        return response

    @staticmethod
    def summarize(video_link, system_message):
        audio_path = YouTubeTranscripter.__download_video(video_link)
        transcripts = YouTubeTranscripter.__generate_scripts_from_audio(audio_path)

        summary_response = ""
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": transcripts}
        ]
        try:
            # Get the chat completion
            from utility.llm import LLM
            client = LLM.get_client()
            
            from langchain_core.output_parsers import StrOutputParser
            parser = StrOutputParser()

            chain = client | parser

            summary_response = chain.invoke(message,
                    config={"temperature": 0, "max_tokens": 500, "model_name": "gpt-4o-mini"})

        except Exception as e:
            print(e) # A better error handling mechanism could be implemented

        return summary_response, transcripts

if __name__ == "__main__":
    # Link to the youtube video
    video_link = "https://www.youtube.com/watch?v=8l8fpR7xMEQ"
    zero_shot_system_message = """You are helpful assistant. Summarize the following youtube transcript in  5-10 lines.
        Keep it concise and make sure to include all the important points. Do mention all the topics covered in the transcript shared
        """
    summary_response, transcripts = YouTubeTranscripter.summarize(video_link, zero_shot_system_message)
    print(f"YouTube Video: {video_link}")
    print("=================================================================================")
    print(f"Summarization using zero-shot-prompting:\n{summary_response}")

    #### Let's evaluate the summary generated by LLM
    summary_rating = YouTubeTranscripter.rate_the_summary(summary_response, transcripts)
    print("---------------------------------------------------------------------------------")
    print(f"Rating of the summerization: {summary_rating}")


    cot_system_message = """
        You are a helpful assistant that summarises youtube transcripts.

        Think step-by-step,

        Read the YouTube transcript, paying close attention to each sentence and its importance.

        Parse the text into meaningful chunks, identifying the main ideas and key points, while ignoring unnecessary details and filler language.

        Extract the most crucial information, such as names, dates, events, or statistics, that capture the essence of the content.

        Consider the overall theme and message the speaker intends to convey, looking for a central idea or argument.

        Begin summarizing by focusing on the main points, using clear and concise language. Ensure the summary maintains the core meaning of the original transcript without unnecessary elaboration.

        Provide a brief introduction and conclusion to bookend the summary, stating the key takeaways and any relevant context the viewer might need.

        Double-check that the summary is coherent, making sense on its own, and that it represents the original transcript truthfully.

        Keep the length reasonable and aligned with the complexity of the content. Aim for a good balance between brevity and inclusivity of essential details.

        Use an engaging tone that suits the summary's purpose and aligns with the original video's intent.

        Finally, review the summary, editing for grammar, clarity, and any potential biases or misunderstandings the concise language might cause. Ensure it's accessible to the intended audience.

    """

    summary_response, transcripts = YouTubeTranscripter.summarize(video_link, cot_system_message)
    print(f"YouTube Video: {video_link}")
    print("=================================================================================")
    print(f"Summarization using cot-prompting:\n{summary_response}")

    #### Let's evaluate the summary generated by LLM
    summary_rating = YouTubeTranscripter.rate_the_summary(summary_response, transcripts)
    print("---------------------------------------------------------------------------------")
    print(f"Rating of the summerization: {summary_rating}")






    