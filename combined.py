import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the text (replace with your own file loading code)
with open("your_file.txt", "r") as file:
    text = file.read()

# Split the text into chunks based on timestamps
time_pattern = r"\d{2}:\d{2}:\d{2}"
chunks = re.split(time_pattern, text)

# Get the timestamps and convert them to datetime objects
times = re.findall(time_pattern, text)
timestamps = [datetime.strptime(time, "%H:%M:%S") for time in times]

# Associate each chunk of text with the timestamp that precedes it
timestamped_text = {timestamps[i]: chunks[i + 1] for i in range(len(timestamps))}

# Initialize a dictionary to store the text for each 10-minute interval
text_intervals = {}

# Group the chunks into 10-minute intervals
start_time = timestamps[0]
end_time = start_time + timedelta(minutes=10)
interval_text = ""

for timestamp in sorted(timestamped_text.keys()):
    if start_time <= timestamp < end_time:
        # This timestamp falls within the current 10-minute interval
        interval_text += timestamped_text[timestamp]
    else:
        # This timestamp falls within the next 10-minute interval
        text_intervals[start_time] = interval_text
        interval_text = timestamped_text[timestamp]
        start_time = end_time
        end_time = start_time + timedelta(minutes=10)

# Add the last interval
text_intervals[start_time] = interval_text

# Initialize the CountVectorizer and LDA models
vectorizer = CountVectorizer(stop_words="english")
lda = LatentDirichletAllocation(n_components=5, random_state=42)

# Perform topic modeling for each interval
topics_intervals = {}

for start_time, text in text_intervals.items():
    # Vectorize the text
    text_vectorized = vectorizer.fit_transform([text])

    # Fit the LDA model
    lda.fit(text_vectorized)

    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    top_words_intervals = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_words_intervals[topic_idx] = [
            feature_names[i] for i in topic.argsort()[: -10 - 1 : -1]
        ]

    # Store the top words for each topic in this interval
    topics_intervals[start_time] = top_words_intervals

# Print the topics for the first few intervals
for start_time, topics in list(topics_intervals.items())[:5]:
    print(f"Interval from {start_time} to {start_time + timedelta(minutes=10)}:")
    for topic_idx, words in topics.items():
        print(f'- Topic {topic_idx}: {", ".join(words)}')
    print()
