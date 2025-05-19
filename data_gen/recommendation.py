import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
import random
from collections import Counter
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class SBERTPlaylistGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the playlist generator with SBERT
        
        Parameters:
        - model_name: The sentence transformer model to use
            'all-MiniLM-L6-v2' - Fast and efficient model with good performance
            'all-mpnet-base-v2' - Higher quality but slower model
        """
        # Database connection
        self.conn = psycopg2.connect(
            host="localhost",
            dbname="youtube_data",
            user="postgres",
            password="postgres"
        )
        self.cursor = self.conn.cursor()
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Load SBERT model
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Data containers
        self.videos_df = None
        self.comments_df = None
        self.video_embeddings = None
        self.video_features = None
        
        # Theme and mood keywords
        self.mood_keywords = {
            'happy': ['happy', 'joy', 'upbeat', 'cheerful', 'fun', 'excited', 'uplifting', 'celebration'],
            'sad': ['sad', 'depression', 'depressing', 'cry', 'crying', 'tears', 'heartbreak', 'melancholy'],
            'energetic': ['energy', 'pump', 'workout', 'gym', 'exercise', 'energetic', 'party', 'dance'],
            'relaxing': ['relax', 'chill', 'peaceful', 'calm', 'sleep', 'study', 'ambient', 'meditation'],
            'nostalgic': ['memory', 'memories', 'nostalgia', 'nostalgic', 'childhood', 'old', 'classic', 'remember'],
            'romantic': ['love', 'romance', 'romantic', 'couple', 'relationship', 'wedding', 'together', 'heart'],
            'focus': ['focus', 'concentrate', 'productivity', 'study', 'work', 'attention', 'deep', 'flow']
        }
        
        # Activity keywords
        self.activity_keywords = {
            'workout': ['gym', 'exercise', 'fitness', 'workout', 'running', 'training', 'cardio', 'muscles'],
            'party': ['party', 'club', 'dance', 'dancing', 'celebrating', 'celebration', 'night out', 'drinks'],
            'study': ['study', 'studying', 'homework', 'concentration', 'focus', 'learning', 'exam', 'school'],
            'travel': ['travel', 'road trip', 'driving', 'journey', 'adventure', 'vacation', 'trip', 'drive'],
            'gaming': ['game', 'gaming', 'play', 'fortnite', 'minecraft', 'xbox', 'playstation', 'streaming']
        }

    def load_data(self):
        """Load videos and comments from the database"""
        print("Loading data from database...")
        
        # Load videos
        self.cursor.execute("""
            SELECT v.video_id, v.title, v.channel_title, v.view_count, v.like_count,
                v.comment_count, v.duration
            FROM videos v
        """)
        
        # Use channel_title as artist
        video_columns = ['video_id', 'title', 'channel_title', 'view_count', 'like_count', 
                        'comment_count', 'duration']
        self.videos_df = pd.DataFrame(self.cursor.fetchall(), columns=video_columns)
        
        # Rename channel_title to artist for consistency in the code
        self.videos_df = self.videos_df.rename(columns={'channel_title': 'artist', 
                                                        'view_count': 'views'})
        
        # Load comments (only top-level comments to reduce noise)
        self.cursor.execute("""
            SELECT c.comment_id, c.video_id, c.text, c.like_count, c.published_at
            FROM comments c
            WHERE c.is_reply = false
            AND length(c.text) BETWEEN 20 AND 1000  -- Filter out very short/long comments
        """)
        
        comment_columns = ['comment_id', 'video_id', 'text', 'like_count', 'published_at']
        self.comments_df = pd.DataFrame(self.cursor.fetchall(), columns=comment_columns)
        
        # Ensure we have quality data
        self.comments_df['text'] = self.comments_df['text'].fillna('')
        self.comments_df = self.comments_df[self.comments_df['text'].str.len() > 20]
        
        print(f"Loaded {len(self.videos_df)} videos and {len(self.comments_df)} comments")
        return self
    def preprocess_comments(self):
        """Clean and preprocess comments"""
        print("Preprocessing comments...")
        
        def clean_text(text):
            """Basic text cleaning"""
            if not isinstance(text, str):
                return ""
                
            # Convert HTML breaks to spaces
            text = re.sub(r'<br>|<br/>', ' ', text)
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^\w\s.,!?]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Clean the comments
        self.comments_df['cleaned_text'] = self.comments_df['text'].apply(clean_text)
        
        # Filter out empty comments after cleaning
        self.comments_df = self.comments_df[self.comments_df['cleaned_text'].str.len() > 20]
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        self.comments_df['sentiment'] = self.comments_df['cleaned_text'].apply(
            lambda text: self.sia.polarity_scores(text)['compound']
        )
        
        # Extract mood and activity indicators
        print("Extracting themes from comments...")
        
        # Combine all keywords for vectorized operations
        all_keywords = {}
        all_keywords.update(self.mood_keywords)
        all_keywords.update(self.activity_keywords)
        
        # Create columns for each mood and activity
        for category, keywords in tqdm(all_keywords.items(), desc="Categorizing comments"):
            self.comments_df[category] = self.comments_df['cleaned_text'].apply(
                lambda text: sum(1 for word in keywords if word.lower() in text.lower())
            )
        
        return self

    def create_embeddings(self, batch_size=32):
        """Create video embeddings using SBERT on aggregated comments"""
        print("Creating video-level features...")
        
        # Aggregate comments by video
        video_comments = self.comments_df.groupby('video_id').agg({
            'cleaned_text': lambda texts: ' '.join(texts),
            'like_count': 'sum',  # Changed from 'likes' to 'like_count'
            'sentiment': 'mean',
            **{cat: 'sum' for cat in list(self.mood_keywords.keys()) + list(self.activity_keywords.keys())}
        }).reset_index()
        
        # Normalize the keyword scores
        categories = list(self.mood_keywords.keys()) + list(self.activity_keywords.keys())
        for cat in categories:
            max_val = video_comments[cat].max()
            if max_val > 0:  # Avoid division by zero
                video_comments[cat] = video_comments[cat] / max_val
        
        # Generate embeddings for each video's comments
        print("Generating SBERT embeddings for video comments...")
        text_to_embed = video_comments['cleaned_text'].tolist()
        
        # Process in batches to avoid memory issues
        embeddings = []
        for i in tqdm(range(0, len(text_to_embed), batch_size), desc="Embedding batches"):
            batch = text_to_embed[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        video_comments['embedding'] = embeddings
        
        # Rename columns to match the expected names in the videos dataframe
        video_comments = video_comments.rename(columns={'like_count': 'comment_likes'})
        
        # Merge with video information
        self.video_features = self.videos_df.merge(
            video_comments, 
            on='video_id', 
            how='inner'  # Only keep videos that have comments
        )
        
        # Extract the embeddings as a numpy array for similarity calculations
        self.video_embeddings = np.array(self.video_features['embedding'].tolist())
        
        print(f"Created embeddings for {len(self.video_features)} videos")
        return self

    def find_similar_videos(self, seed_idx, n=10, exclude_seed=True):
        """Find videos similar to the seed video based on comment embeddings"""
        # Get the embedding of the seed video
        seed_embedding = self.video_embeddings[seed_idx].reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(seed_embedding, self.video_embeddings)[0]
        
        # Sort by similarity (highest first)
        if exclude_seed:
            # Set the seed's similarity to -1 to exclude it
            similarities[seed_idx] = -1
            
        # Get top indices
        top_indices = similarities.argsort()[::-1][:n]
        
        return top_indices, similarities[top_indices]

    def generate_playlist(self, mode='popular', seed_video_id=None, mood=None, activity=None, size=10):
        """
        Generate a playlist based on various criteria
        
        Parameters:
        - mode: Method to generate playlist
            'popular': Most popular videos (default)
            'seed': Based on similarity to a seed video
            'mood': Based on a specific mood
            'activity': Based on a specific activity
            'random': Random selection
        - seed_video_id: Video ID to use as seed (required if mode='seed')
        - mood: Mood to base playlist on (required if mode='mood')
        - activity: Activity to base playlist on (required if mode='activity')
        - size: Number of videos to include in playlist
        
        Returns:
        - List of dicts with video information
        """
        if not hasattr(self, 'video_features') or len(self.video_features) == 0:
            print("Features not created. Run load_data().preprocess_comments().create_embeddings() first.")
            return []
        
        if mode == 'seed' and seed_video_id:
            return self._generate_from_seed(seed_video_id, size)
        elif mode == 'mood' and mood:
            return self._generate_from_mood(mood, size)
        elif mode == 'activity' and activity:
            return self._generate_from_activity(activity, size)
        elif mode == 'random':
            return self._generate_random(size)
        else:
            return self._generate_popular(size)
    
    def _generate_from_seed(self, seed_video_id, size=10):
        """Generate playlist based on similarity to seed video"""
        print(f"Generating playlist based on seed video: {seed_video_id}")
        
        # Find the index of the seed video
        seed_videos = self.video_features[self.video_features['video_id'] == seed_video_id]
        
        if len(seed_videos) == 0:
            print(f"Seed video {seed_video_id} not found. Generating popular playlist instead.")
            return self._generate_popular(size)
        
        seed_idx = seed_videos.index[0]
        seed_info = seed_videos.iloc[0]
        
        # Find similar videos
        similar_indices, similarities = self.find_similar_videos(seed_idx, n=size, exclude_seed=True)
        
        # Create the playlist
        playlist = []
        for idx, sim in zip(similar_indices, similarities):
            video = self.video_features.iloc[idx]
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'similarity': float(sim)
            })
        
        print(f"Generated playlist of {len(playlist)} videos similar to '{seed_info['title']}'")
        return playlist
    
    def _generate_from_mood(self, mood, size=10):
        """Generate playlist based on a specific mood"""
        valid_moods = list(self.mood_keywords.keys())
        
        if mood not in valid_moods:
            print(f"Invalid mood '{mood}'. Choose from: {valid_moods}")
            return self._generate_popular(size)
        
        print(f"Generating '{mood}' playlist")
        
        # Sort videos by mood score and sentiment appropriately
        if mood in ['happy', 'energetic']:
            # For positive moods, prefer positive sentiment
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'sentiment', 'like_count'], 
                ascending=[False, False, False]
            )
        elif mood in ['sad']:
            # For sad mood, prefer negative sentiment
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'sentiment', 'like_count'], 
                ascending=[False, True, False]
            )
        else:
            # For neutral moods, sort just by the mood score
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'views'], 
                ascending=[False, False]
            )
        
        # Take the top videos
        top_videos = sorted_videos.head(size)
        
        # Create the playlist
        playlist = []
        for _, video in top_videos.iterrows():
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'mood_score': float(video[mood])
            })
        
        print(f"Generated '{mood}' playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_from_activity(self, activity, size=10):
        """Generate playlist based on a specific activity"""
        valid_activities = list(self.activity_keywords.keys())
        
        if activity not in valid_activities:
            print(f"Invalid activity '{activity}'. Choose from: {valid_activities}")
            return self._generate_popular(size)
        
        print(f"Generating '{activity}' playlist")
        
        # Sort videos by activity score
        sorted_videos = self.video_features.sort_values(
            by=[activity, 'views'], 
            ascending=[False, False]
        )
        
        # Take the top videos
        top_videos = sorted_videos.head(size)
        
        # Create the playlist
        playlist = []
        for _, video in top_videos.iterrows():
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'activity_score': float(video[activity])
            })
        
        print(f"Generated '{activity}' playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_popular(self, size=10):
        """Generate playlist based on popularity"""
        print("Generating popular playlist")
        
        # Sort by views
        sorted_videos = self.video_features.sort_values(by='views', ascending=False)
        top_videos = sorted_videos.head(size)
        
        # Create the playlist
        playlist = []
        for _, video in top_videos.iterrows():
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'views': int(video['views'])
            })
        
        print(f"Generated popular playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_random(self, size=10):
        """Generate a random playlist"""
        print("Generating random playlist")
        
        # Get random indices
        random_indices = random.sample(range(len(self.video_features)), min(size, len(self.video_features)))
        
        # Create the playlist
        playlist = []
        for idx in random_indices:
            video = self.video_features.iloc[idx]
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist']
            })
        
        print(f"Generated random playlist with {len(playlist)} videos")
        return playlist

    def get_video_info(self, video_id):
        """Get detailed information about a specific video"""
        if video_id not in self.video_features['video_id'].values:
            return None
            
        # Get video details
        video = self.video_features[self.video_features['video_id'] == video_id].iloc[0]
        
        # Get top comments for this video
        video_comments = self.comments_df[self.comments_df['video_id'] == video_id]
        top_comments = video_comments.sort_values(by='like_count', ascending=False).head(5)
        
        # Calculate most common words in comments
        all_text = ' '.join(video_comments['cleaned_text'])
        words = re.findall(r'\b\w+\b', all_text.lower())
        common_words = Counter(words).most_common(10)
        
        # Get mood profile
        mood_profile = {mood: float(video[mood]) for mood in self.mood_keywords}
        activity_profile = {activity: float(video[activity]) for activity in self.activity_keywords}
        
        return {
            'video_id': video_id,
            'title': video['title'],
            'artist': video['artist'],
            'views': int(video['views']),
            'likes': int(video['like_count']),
            'sentiment': float(video['sentiment']),
            'mood_profile': mood_profile,
            'activity_profile': activity_profile,
            'top_comments': top_comments[['text', 'like_count']].values.tolist(),
            'common_words': common_words
        }
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Database connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize and load data
    recommender = SBERTPlaylistGenerator()
    recommender.load_data().preprocess_comments().create_embeddings()
    
    # Generate different types of playlists
    popular_playlist = recommender.generate_playlist(mode='popular', size=5)
    print("\nPopular Playlist:")
    for i, video in enumerate(popular_playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} ({video['views']} views)")
    
    # Generate mood-based playlist
    happy_playlist = recommender.generate_playlist(mode='mood', mood='happy', size=5)
    print("\nHappy Playlist:")
    for i, video in enumerate(happy_playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} (mood score: {video['mood_score']:.2f})")
    
    # Generate activity-based playlist
    workout_playlist = recommender.generate_playlist(mode='activity', activity='workout', size=5)
    print("\nWorkout Playlist:")
    for i, video in enumerate(workout_playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} (activity score: {video['activity_score']:.2f})")
    
    # If you have a seed video ID, generate similar playlist
    if len(popular_playlist) > 0:
        seed_id = popular_playlist[0]['video_id']
        similar_playlist = recommender.generate_playlist(mode='seed', seed_video_id=seed_id, size=5)
        print(f"\nSimilar to '{popular_playlist[0]['title']}':")
        for i, video in enumerate(similar_playlist, 1):
            print(f"{i}. {video['title']} by {video['artist']} (similarity: {video['similarity']:.2f})")
    
    # Close connection
    recommender.close()