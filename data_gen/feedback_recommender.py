import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class FeedbackEnhancedRecommender:
    """
    Enhanced music recommender that incorporates user feedback and preference learning
    to continuously improve recommendations over multiple sessions.
    """
    
    def __init__(self, base_recommender):
        """
        Initialize the feedback-enhanced recommender
        
        Parameters:
        - base_recommender: An instance of EnhancedMusicRecommender
        """
        self.recommender = base_recommender
        self.user_data_dir = './user_data'
        os.makedirs(self.user_data_dir, exist_ok=True)
        
        # User session history and preferences
        self.user_history = {}
        self.user_preferences = {}
        self.session_recommendations = {}
        self.current_user_id = None
        self.current_session_id = None
        
        # Recommendation diversity parameters
        self.exploration_ratio = 0.2  # 20% exploration, 80% exploitation
        self.genre_diversity_weight = 0.3
        self.novelty_weight = 0.2
        
        # Feature weights for personalization (will be adjusted based on feedback)
        self.feature_weights = {
            'mood': 1.0,
            'activity': 1.0,
            'genre': 1.0,
            'sentiment': 0.7,
            'theme': 0.5,
            'popularity': 0.3
        }
        
        # Initialize feedback history
        self.feedback_history = {}
        
        # Add XGBoost recommender
        self.xgboost_recommender = XGBoostRecommender(self)
        self.use_xgboost = False  # Flag to control when to use XGBoost

    def get_or_create_user(self, user_id=None, user_name=None, create_if_not_exists=True):
        """
        Get a user by ID or create a new user
        
        Parameters:
        - user_id: Optional user ID
        - user_name: Optional user name for new users
        - create_if_not_exists: Whether to create a new user if not found
        
        Returns:
        - user_id: The user ID
        """
        if user_id is None and not create_if_not_exists:
            raise ValueError("User ID is required when create_if_not_exists=False")
            
        # If user_id provided, try to load existing user
        if user_id is not None:
            user_file = os.path.join(self.user_data_dir, f'user_{user_id}.json')
            if os.path.exists(user_file):
                try:
                    with open(user_file, 'r') as f:
                        user_data = json.load(f)
                    self.user_history[user_id] = user_data.get('history', [])
                    self.user_preferences[user_id] = user_data.get('preferences', {})
                    self.feedback_history[user_id] = user_data.get('feedback', [])
                    self.current_user_id = user_id
                    print(f"Loaded user data for user_id: {user_id}")
                    return user_id
                except Exception as e:
                    print(f"Error loading user data: {e}")
                    if not create_if_not_exists:
                        raise
        
        # Create new user if needed
        if create_if_not_exists:
            # Generate a new user ID if none provided
            if user_id is None:
                user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Initialize user data
            self.user_history[user_id] = []
            self.user_preferences[user_id] = {
                'favorite_genres': {},
                'favorite_moods': {},
                'favorite_activities': {},
                'disliked_genres': {},
                'preferred_eras': {},
                'favorite_artists': {}
            }
            self.feedback_history[user_id] = []
            
            # Save initial user data
            self._save_user_data(user_id, user_name)
            self.current_user_id = user_id
            print(f"Created new user with ID: {user_id}")
            return user_id
        
        raise ValueError(f"User {user_id} not found and create_if_not_exists=False")

    def process_user_input(self, user_id, text_input, additional_context=None):
        """
        Process user input to start a new recommendation session
        
        Parameters:
        - user_id: User ID
        - text_input: User's blog post or text describing their preferences
        - additional_context: Optional dictionary with more structured input
                             (e.g., {'explicit_genre': 'Rock', 'mood': 'relaxed'})
        
        Returns:
        - session_id: ID of the new session
        """
        # Ensure user exists
        if user_id not in self.user_history:
            self.get_or_create_user(user_id)
        
        self.current_user_id = user_id
        
        # Create new session
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        self.current_session_id = session_id
        
        # Store session data
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'text_input': text_input,
            'additional_context': additional_context or {},
            'recommendations': [],
            'feedback': {},
            'completed': False
        }
        
        # Add to user history
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append({'session_id': session_id, 'data': session_data})
        
        # Save immediately to persist
        self._save_user_data(user_id)
        
        # Store session recommendations (will be populated later)
        self.session_recommendations[session_id] = []
        
        return session_id

    def generate_recommendations(self, session_id=None, batch_size=5, strategy='diverse'):
        """
        Generate recommendations using XGBoost if available
        """
        print("\n=== Generating Recommendations Debug ===")
        print(f"Using XGBoost: {self.use_xgboost}")
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No active session. Call process_user_input first.")
        
        user_id = self.current_user_id
        
        # Find session data
        session_data = None
        for session_entry in self.user_history[user_id]:
            if session_entry['session_id'] == session_id:
                session_data = session_entry['data']
                break
        
        if session_data is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")
        
        # Get previously recommended videos to avoid duplicates
        previous_recommendations = set()
        for session_entry in self.user_history[user_id]:
            for rec in session_entry['data'].get('recommendations', []):
                previous_recommendations.add(rec['video_id'])
        
        # Also check current session's recommendations
        for rec in self.session_recommendations.get(session_id, []):
            previous_recommendations.add(rec['video_id'])
        
        # Determine diversity level based on strategy
        diversity_level = {
            'diverse': 0.7,
            'focused': 0.2,
            'novel': 0.9,
            'popular': 0.3
        }.get(strategy, 0.5)
        
        # Mix of strategies based on user history and preferences
        recommendation_methods = []
        
        # Analyze user input text
        text_input = session_data['text_input']
        
        # Get the user's preferences
        preferences = self.user_preferences.get(user_id, {})
        favorite_genres = preferences.get('favorite_genres', {})
        favorite_moods = preferences.get('favorite_moods', {})
        
        # Determine appropriate mix of recommendation methods
        # 1. Always include text-based as primary method
        recommendation_methods.append(('text', {
            'user_text': text_input,
            'size': int(batch_size * 0.5),  # 60% from text analysis
            'diversity': diversity_level
        }))
        
        # 2. Add mood-based if we have favorite moods
        if favorite_moods and random.random() < 0.7:  # 70% chance
            top_mood = max(favorite_moods.items(), key=lambda x: x[1])[0] if favorite_moods else None
            if top_mood:
                recommendation_methods.append(('mood', {
                    'mood': top_mood,
                    'size': int(batch_size * 0.2),  # 20% from favorite mood
                    'diversity': diversity_level
                }))
        
        # 3. Add genre mix if we have favorite genres
        if favorite_genres and random.random() < 0.6:  # 60% chance
            # Convert to proportions
            total = sum(favorite_genres.values())
            if total > 0:  # Only proceed if we have valid genre scores
                genre_proportions = {g: min(0.8, v/total) for g, v in favorite_genres.items() if v > 0}
                
                # Make sure proportions sum to 1
                total_prop = sum(genre_proportions.values())
                if total_prop > 0:
                    genre_proportions = {g: v/total_prop for g, v in genre_proportions.items()}
                    
                    recommendation_methods.append(('genre_mix', {
                        'genres': genre_proportions,
                        'size': int(batch_size * 0.3),  # 20% from favorite genres
                        'mood': None  # Could set this based on text analysis
                    }))
            else:
                # If no valid genre scores, add some popular recommendations instead
                recommendation_methods.append(('popular', {
                    'size': int(batch_size * 0.3)  # 30% popular songs
                }))
        
        # 4. Add some popular recommendations for new users
        if len(self.user_history[user_id]) <= 3:  # New user with few sessions
            recommendation_methods.append(('popular', {
                'size': int(batch_size * 0.2)  # 20% popular songs
            }))
            
        # 5. If we need more methods to reach batch_size
        remaining = batch_size - sum(m[1].get('size', 0) for m in recommendation_methods)
        if remaining > 0:
            # Add some random recommendations for exploration
            recommendation_methods.append(('random', {
                'size': remaining
            }))
        
        # Execute each recommendation method
        all_recommendations = []
        
        for method, params in recommendation_methods:
            if method == 'text':
                recs = self.recommender.generate_from_text(
                    params['user_text'], 
                    params['size'], 
                    params['diversity']
                )
            elif method == 'mood':
                recs = self.recommender.generate_playlist(
                    mode='mood',
                    mood=params['mood'],
                    size=params['size'],
                    diversity=params['diversity']
                )
            elif method == 'genre_mix':
                recs = self.recommender.generate_genre_mix(
                    genres=params['genres'],
                    size=params['size'],
                    mood=params['mood']
                )
            elif method == 'popular':
                recs = self.recommender.generate_playlist(
                    mode='popular',
                    size=params['size']
                )
            elif method == 'random':
                recs = self.recommender.generate_playlist(
                    mode='random',
                    size=params['size']
                )
            else:
                # Default to text-based
                recs = self.recommender.generate_from_text(text_input, params['size'])
            
            # Add the method info to each recommendation
            for rec in recs:
                rec['recommendation_method'] = method
            
            all_recommendations.extend(recs)
        
        # Filter out previously recommended videos
        filtered_recommendations = [
            rec for rec in all_recommendations 
            if rec['video_id'] not in previous_recommendations
        ]
        
        # If we don't have enough after filtering, get more
        if len(filtered_recommendations) < batch_size:
            # Calculate how many more we need
            needed = batch_size - len(filtered_recommendations)
            
            # Get additional recommendations with explicit exclusions
            additional_recs = self.recommender.generate_from_text(
                text_input, 
                size=needed * 2,  # Get extra to account for further filtering
                diversity=diversity_level + 0.2  # Increase diversity for more variety
            )
            
            # Filter these as well
            additional_filtered = [
                rec for rec in additional_recs
                if rec['video_id'] not in previous_recommendations and
                   rec['video_id'] not in [r['video_id'] for r in filtered_recommendations]
            ]
            
            # Add to our recommendations
            filtered_recommendations.extend(additional_filtered[:needed])
        
        # Select final set to exactly match batch_size
        final_recommendations = filtered_recommendations[:batch_size]
        
        # Apply personalized re-ranking based on user preferences
        final_recommendations = self._personalized_ranking(final_recommendations, user_id)
        
        # Add artist-based recommendations if user has favorite artists
        if user_id and 'favorite_artists' in self.user_preferences[user_id]:
            favorite_artists = self.user_preferences[user_id]['favorite_artists']
            if favorite_artists:
                print(f"Adding recommendations from favorite artists: {favorite_artists}")
                artist_recommendations = []
                for artist in favorite_artists:
                    # Find videos from this artist
                    artist_videos = [
                        v for v in self.user_history[user_id]
                        if v['data'].get('artist', '').lower() == artist.lower()
                        and v['data'].get('video_id') not in [r['video_id'] for r in final_recommendations]
                    ]
                    # Add up to 2 videos per artist
                    artist_recommendations.extend(artist_videos[:2])
                
                # Add artist recommendations to the pool
                if artist_recommendations:
                    final_recommendations.extend(artist_recommendations)
                    print(f"Added {len(artist_recommendations)} artist-based recommendations")
        
        # If XGBoost is available, use it to re-rank recommendations
        if self.use_xgboost:
            print("\nUsing XGBoost model for recommendation re-ranking...")
            # Get predicted ratings from XGBoost
            predicted_ratings = self.xgboost_recommender.predict_ratings(final_recommendations)
            
            # Combine recommendations with predictions
            scored_recommendations = list(zip(final_recommendations, predicted_ratings))
            
            # Sort by predicted rating
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Take top batch_size recommendations
            final_recommendations = [rec[0] for rec in scored_recommendations[:batch_size]]
            print("XGBoost re-ranking complete!")
        else:
            print("\nUsing base recommendation system (no XGBoost)...")
            print("Current feedback count:", len(self.feedback_history.get(user_id, [])))
            print(f"Need {self.xgboost_recommender.min_feedback_threshold} feedback points to enable XGBoost")
        
        # Store recommendations in session
        self.session_recommendations[session_id].extend(final_recommendations)
        
        # Update session data
        for session_entry in self.user_history[user_id]:
            if session_entry['session_id'] == session_id:
                session_entry['data']['recommendations'].extend(final_recommendations)
                break
        
        # Save user data
        self._save_user_data(user_id)
        
        return final_recommendations

    def record_feedback(self, session_id, video_id, rating, skip_reason=None, listen_duration=None):
        """
        Record user feedback for a video
        """
        print("\n=== Recording Feedback Debug ===")
        print(f"Recording feedback for video {video_id} with rating {rating}")
        print(f"Current user ID: {self.current_user_id}")
        print(f"Session ID: {session_id}")
        
        if not self.current_user_id:
            print("ERROR: No active user!")
            raise ValueError("No active user")
            
        # Find video in session recommendations or history
        video_data = None
        print("Searching for video data...")
        
        # First check in session recommendations
        if session_id in self.session_recommendations:
            print("Checking session recommendations...")
            session_data = self.session_recommendations[session_id]
            if isinstance(session_data, dict) and 'recommendations' in session_data:
                for rec in session_data['recommendations']:
                    if rec['video_id'] == video_id:
                        video_data = rec
                        print("Found video in session recommendations")
                        break
                
        # If not found, check in user history
        if not video_data:
            print("Checking user history...")
            for session_entry in self.user_history[self.current_user_id]:
                if session_entry['session_id'] == session_id:
                    for rec in session_entry['data'].get('recommendations', []):
                        if rec['video_id'] == video_id:
                            video_data = rec
                            print("Found video in user history")
                            break
                    break
        
        if not video_data:
            print(f"ERROR: Video {video_id} not found in session or history")
            raise ValueError(f"Video {video_id} not found in session or history")
            
        # Create feedback data
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'video_id': video_id,
            'rating': rating,
            'skip_reason': skip_reason,
            'listen_duration': listen_duration,
            'video_data': video_data
        }
        
        print(f"Created feedback data: {feedback}")
        
        # Extract artist preference from skip_reason if it's a description
        if skip_reason and isinstance(skip_reason, str) and len(skip_reason) > 10:
            # Check for artist mentions in the description
            artist = video_data.get('artist', '')
            if artist and artist.lower() in skip_reason.lower():
                # Add artist to user preferences
                if 'favorite_artists' not in self.user_preferences[self.current_user_id]:
                    self.user_preferences[self.current_user_id]['favorite_artists'] = []
                if artist not in self.user_preferences[self.current_user_id]['favorite_artists']:
                    self.user_preferences[self.current_user_id]['favorite_artists'].append(artist)
                    print(f"Added {artist} to favorite artists based on feedback")
        
        # Add to user's feedback history
        if self.current_user_id not in self.feedback_history:
            self.feedback_history[self.current_user_id] = []
        self.feedback_history[self.current_user_id].append(feedback)
        print(f"Added feedback to history. Total feedback count: {len(self.feedback_history[self.current_user_id])}")
        
        # Update session data
        if session_id in self.session_recommendations:
            session_data = self.session_recommendations[session_id]
            if isinstance(session_data, dict):
                if 'feedback' not in session_data:
                    session_data['feedback'] = []
                session_data['feedback'].append(feedback)
                print("Updated session data with feedback")
        
        # Update user preferences
        print("Updating user preferences...")
        self._update_user_preferences(self.current_user_id, video_data, rating)
        
        # Save user data
        print("Saving user data...")
        self._save_user_data(self.current_user_id)
        
        # Adjust feature weights
        print("Adjusting feature weights...")
        self._adjust_feature_weights(self.current_user_id)
        
        # Try to train XGBoost model if enough feedback
        print("\nAttempting to train XGBoost model...")
        try:
            if self.xgboost_recommender.train_model(self.current_user_id):
                self.use_xgboost = True
                print("XGBoost model trained successfully!")
                
                # Print current feedback count and distribution
                feedback_count = len(self.feedback_history[self.current_user_id])
                print(f"\nCurrent feedback count: {feedback_count}")
                
                # Calculate rating distribution
                ratings = [f.get('rating', 0) for f in self.feedback_history[self.current_user_id]]
                rating_dist = {}
                for r in ratings:
                    rating_dist[r] = rating_dist.get(r, 0) + 1
                
                print("\nRating Distribution:")
                for rating in sorted(rating_dist.keys()):
                    print(f"{rating} stars: {rating_dist[rating]} songs")
                
            else:
                print("XGBoost model training failed - not enough feedback")
                print(f"Current feedback count: {len(self.feedback_history[self.current_user_id])}")
                print(f"Need {self.xgboost_recommender.min_feedback_threshold} feedback points to enable XGBoost")
        except Exception as e:
            print(f"Error training XGBoost model: {str(e)}")
            self.use_xgboost = False
            
        return feedback

    def generate_final_playlist(self, user_id, name=None, size=20, description=None):
        """
        Generate a 'best of' playlist based on user feedback history
        
        Parameters:
        - user_id: User ID
        - name: Optional playlist name
        - size: Number of songs in the playlist
        - description: Optional playlist description
        
        Returns:
        - playlist: Final playlist based on learned preferences
        """
        # Set current user
        self.current_user_id = user_id
        
        # Ensure user exists
        if user_id not in self.user_history:
            raise ValueError(f"User {user_id} not found")
        
        # Get user preferences
        preferences = self.user_preferences.get(user_id, {})
        
        # Get feedback history
        feedback = self.feedback_history.get(user_id, [])
        
        if not feedback:
            # No feedback, generate based on general preferences
            if not name:
                name = "Your Personalized Playlist"
            if not description:
                description = "A playlist created just for you based on your preferences."
                
            # Use the most recent session text if available
            if self.user_history[user_id]:
                latest_session = max(self.user_history[user_id], 
                                    key=lambda x: x['data'].get('timestamp', ''))
                text_input = latest_session['data'].get('text_input', '')
                
                # Generate from the latest text input
                playlist = self.recommender.generate_from_text(
                    text_input, 
                    size=size,
                    diversity=0.6  # Good balance between cohesion and variety
                )
                
                return {
                    'name': name,
                    'description': description,
                    'created_at': datetime.now().isoformat(),
                    'size': len(playlist),
                    'tracks': playlist,
                    'method': 'text_based'
                }
            else:
                # No sessions either, use popular recommendations
                playlist = self.recommender.generate_playlist(
                    mode='popular',
                    size=size
                )
                
                return {
                    'name': name or "Popular Tracks You Might Enjoy",
                    'description': description or "A selection of popular tracks to get you started.",
                    'created_at': datetime.now().isoformat(),
                    'size': len(playlist),
                    'tracks': playlist,
                    'method': 'popular'
                }
        
        # Find highly rated songs
        highly_rated = [f for f in feedback if f.get('rating', 0) >= 4]
        
        # If we don't have enough highly rated songs, include songs rated 3+
        if len(highly_rated) < 5:
            highly_rated = [f for f in feedback if f.get('rating', 0) >= 3]
        
        # Create a candidate pool from highly rated songs
        candidate_pool = []
        for feedback_item in highly_rated:
            video_data = feedback_item.get('video_data', {})
            if video_data and 'video_id' in video_data:
                # Add rating to the video data
                video_data['user_rating'] = feedback_item.get('rating', 0)
                candidate_pool.append(video_data)
        
        # Calculate genre distribution
        genre_counts = Counter([v.get('genre', 'Unknown') for v in candidate_pool 
                               if v.get('genre') != 'Unknown'])
        total_genres = sum(genre_counts.values())
        
        if total_genres > 0:
            genre_distribution = {genre: count/total_genres for genre, count in genre_counts.items()}
        else:
            # Default to some popular genres if we don't have genre data
            genre_distribution = {'Pop': 0.4, 'Rock': 0.3, 'Hip-Hop': 0.3}
        
        # Include the user's favorite genres from preferences
        favorite_genres = preferences.get('favorite_genres', {})
        if favorite_genres:
            # Normalize to proportions
            total = sum(favorite_genres.values())
            if total > 0:
                for genre, score in favorite_genres.items():
                    # Blend with existing distribution (70% from feedback, 30% from preferences)
                    if genre in genre_distribution:
                        genre_distribution[genre] = 0.7 * genre_distribution.get(genre, 0) + 0.3 * (score/total)
                    else:
                        genre_distribution[genre] = 0.3 * (score/total)
        
        # We want some of the user's favorite songs plus recommendations based on them
        direct_picks = min(size // 4, len(candidate_pool))  # 25% direct picks
        
        # Sort candidate pool by rating and add variety
        candidate_pool.sort(key=lambda x: (
            x.get('user_rating', 0),  # First by rating
            random.random()  # Then randomly to break ties
        ), reverse=True)
        
        # Take the top rated songs for direct inclusion
        direct_selections = candidate_pool[:direct_picks]
        
        # Use the remaining slots for recommendations based on these selections
        remaining_slots = size - len(direct_selections)
        
        # Generate similar songs for each direct pick
        similar_songs = []
        seed_ids_used = set()
        
        # Use each direct pick as a seed for similar songs
        for video in direct_selections:
            # Skip if we've already used this as a seed
            if video['video_id'] in seed_ids_used:
                continue
                
            # Generate similar songs
            similar = self.recommender.generate_playlist(
                mode='seed',
                seed_video_id=video['video_id'],
                size=max(2, remaining_slots // len(direct_selections)),
                diversity=0.4  # More focused on similarity for cohesion
            )
            
            # Mark this as used
            seed_ids_used.add(video['video_id'])
            
            # Add to our pool of similar songs
            similar_songs.extend(similar)
        
        # If we need more songs, add some based on genre distribution
        if len(direct_selections) + len(similar_songs) < size:
            remaining = size - len(direct_selections) - len(similar_songs)
            
            # Generate genre-based recommendations
            genre_recs = self.recommender.generate_genre_mix(
                genres=genre_distribution,
                size=remaining
            )
            
            similar_songs.extend(genre_recs)
        
        # Make sure we don't have duplicates
        used_ids = set(v['video_id'] for v in direct_selections)
        unique_similar = []
        
        for video in similar_songs:
            if video['video_id'] not in used_ids:
                unique_similar.append(video)
                used_ids.add(video['video_id'])
                
                if len(direct_selections) + len(unique_similar) >= size:
                    break
        
        # Combine direct selections and similar songs
        final_playlist = direct_selections + unique_similar[:size - len(direct_selections)]
        
        # Shuffle to mix direct picks and recommendations
        random.shuffle(final_playlist)
        
        # Create the playlist metadata
        if not name:
            # Generate a name based on the content
            genres = list(genre_distribution.keys())
            if len(genres) >= 2:
                name = f"Your {genres[0]} & {genres[1]} Mix"
            elif len(genres) == 1:
                name = f"Your {genres[0]} Favorites"
            else:
                name = "Your Personal Favorites"
                
        if not description:
            description = "A personalized playlist based on your listening history and preferences."
            
        playlist_data = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'size': len(final_playlist),
            'tracks': final_playlist,
            'genre_distribution': genre_distribution,
            'direct_picks': len(direct_selections),
            'method': 'feedback_based'
        }
        
        # Save this playlist in user data
        if 'playlists' not in self.user_preferences[user_id]:
            self.user_preferences[user_id]['playlists'] = []
            
        playlist_id = f"playlist_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        self.user_preferences[user_id]['playlists'].append({
            'playlist_id': playlist_id,
            'data': playlist_data
        })
        
        # Save user data
        self._save_user_data(user_id)
        
        return playlist_data

    def _personalized_ranking(self, recommendations, user_id):
        """
        Apply personalized ranking to recommendations based on user preferences
        
        Parameters:
        - recommendations: List of recommended videos
        - user_id: User ID
        
        Returns:
        - reranked_recommendations: Reordered recommendations
        """
        if not recommendations:
            return []
            
        # Get user preferences
        preferences = self.user_preferences.get(user_id, {})
        
        # Score each recommendation based on user preferences
        scored_recs = []
        
        for rec in recommendations:
            # Start with the base score (e.g., match_score)
            base_score = rec.get('match_score', 0)
            if base_score == 0:
                # If no match score, use a default value
                base_score = 0.5
            
            # Initialize personalized score
            personalized_score = base_score
            
            # Adjust score based on genre preferences
            genre = rec.get('genre')
            if genre and 'favorite_genres' in preferences:
                genre_factor = preferences['favorite_genres'].get(genre, 0) * self.feature_weights['genre']
                personalized_score += genre_factor * 0.2
            
            # Adjust score based on artist preferences
            artist = rec.get('artist')
            if artist and 'favorite_artists' in preferences:
                artist_factor = preferences['favorite_artists'].get(artist, 0) * 2.0  # Strong weight for artists
                personalized_score += artist_factor * 0.3
            
            # Adjust for novelty - slightly boost videos with fewer views
            if 'views' in rec:
                views = rec['views']
                # Logarithmic scaling to normalize view counts
                log_views = np.log1p(views) if views > 0 else 0
                # Invert so that fewer views gives higher score
                novelty_score = max(0, 1 - (log_views / 20))  # 20 is approx log(e^20) ≈ 500M views
                personalized_score += novelty_score * self.novelty_weight * 0.1
            
            # Store the scored recommendation
            scored_recs.append((personalized_score, rec))
        
        # Sort by personalized score (descending)
        scored_recs.sort(key=lambda x: x[0], reverse=True)  # Sort by the score, not the recommendation
        
        # Extract just the recommendations in new order
        reranked_recommendations = [rec for _, rec in scored_recs]
        
        return reranked_recommendations

    def _update_user_preferences(self, user_id, video_data, rating):
        """
        Update user preferences based on feedback
        
        Parameters:
        - user_id: User ID
        - video_data: Video data that was rated
        - rating: Rating (1-5 scale)
        """
        # Initialize preferences if needed
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'favorite_genres': {},
                'favorite_moods': {},
                'favorite_activities': {},
                'disliked_genres': {},
                'preferred_eras': {},
                'favorite_artists': {}
            }
        
        preferences = self.user_preferences[user_id]
        
        # Calculate the preference update value based on rating
        if rating >= 4:  # Highly rated
            update_value = (rating - 3) / 2  # Maps 4->0.5, 5->1.0
        elif rating <= 2:  # Disliked
            update_value = -((3 - rating) / 2)  # Maps 2->-0.5, 1->-1.0
        else:  # Neutral
            update_value = 0.1  # Small positive for exposure
        
        # Update genre preferences
        genre = video_data.get('genre')
        if genre:
            # Update favorite genres
            if 'favorite_genres' not in preferences:
                preferences['favorite_genres'] = {}
                
            current_score = preferences['favorite_genres'].get(genre, 0)
            
            # Update with smoothing to avoid wild swings
            new_score = current_score + update_value
            
            # Keep scores bounded
            new_score = max(-1.0, min(5.0, new_score))
            
            preferences['favorite_genres'][genre] = new_score
            
            # If disliked, also add to disliked_genres
            if update_value < 0:
                if 'disliked_genres' not in preferences:
                    preferences['disliked_genres'] = {}
                preferences['disliked_genres'][genre] = preferences['disliked_genres'].get(genre, 0) - update_value
        
        # Update artist preferences
        artist = video_data.get('artist')
        if artist:
            if 'favorite_artists' not in preferences:
                preferences['favorite_artists'] = {}
                
            current_score = preferences['favorite_artists'].get(artist, 0)
            
            # Artists get bigger updates (more specific preference)
            artist_update = update_value * 1.5
            
            # Update with smoothing
            new_score = current_score + artist_update
            
            # Keep scores bounded
            new_score = max(-1.0, min(5.0, new_score))
            
            preferences['favorite_artists'][artist] = new_score
        
        # Update mood preferences if available
        for mood in self.recommender.mood_keywords:
            if hasattr(video_data, mood) or mood in video_data:
                mood_value = video_data.get(mood, 0)
                if mood_value > 0.3:  # Only count significant mood associations
                    if 'favorite_moods' not in preferences:
                        preferences['favorite_moods'] = {}
                    
                    current_score = preferences['favorite_moods'].get(mood, 0)
                    
                    # Weight update by mood relevance and rating
                    mood_update = update_value * mood_value
                    
                    # Update score
                    new_score = current_score + mood_update
                    
                    # Keep scores bounded
                    new_score = max(-1.0, min(5.0, new_score))
                    
                    preferences['favorite_moods'][mood] = new_score
        
        # Update activity preferences if available
        for activity in self.recommender.activity_keywords:
            if hasattr(video_data, activity) or activity in video_data:
                activity_value = video_data.get(activity, 0)
                if activity_value > 0.3:  # Only count significant activity associations
                    if 'favorite_activities' not in preferences:
                        preferences['favorite_activities'] = {}
                    
                    current_score = preferences['favorite_activities'].get(activity, 0)
                    
                    # Weight update by activity relevance and rating
                    activity_update = update_value * activity_value
                    
                    # Update score
                    new_score = current_score + activity_update
                    
                    # Keep scores bounded
                    new_score = max(-1.0, min(5.0, new_score))
                    
                    preferences['favorite_activities'][activity] = new_score

        # Clean up preferences by removing near-zero entries
        for category in preferences:
            if isinstance(preferences[category], dict):
                # Create a new dict with only significant preferences
                preferences[category] = {k: v for k, v in preferences[category].items() if abs(v) > 0.1}

    def _adjust_feature_weights(self, user_id):
        """
        Adjust feature weights based on feedback history
        
        Parameters:
        - user_id: User ID
        """
        feedback = self.feedback_history.get(user_id, [])
        
        if len(feedback) < 5:
            # Not enough feedback to make meaningful adjustments
            return
        
        # Analyze recent feedback to determine which features correlate with high ratings
        recent_feedback = sorted(feedback, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
        
        # Initialize correlation trackers
        feature_correlations = {
            'mood': [],
            'activity': [],
            'genre': [],
            'sentiment': [],
            'theme': [],
            'popularity': []
        }
        
        # Calculate correlation between features and ratings
        for feedback_item in recent_feedback:
            rating = feedback_item.get('rating', 3)  # Default to neutral
            video_data = feedback_item.get('video_data', {})
            
            if not video_data:
                continue
            
            # Check mood correlations
            mood_score = 0
            for mood in self.recommender.mood_keywords:
                if mood in video_data:
                    mood_value = video_data.get(mood, 0)
                    if mood_value > 0:
                        mood_score = max(mood_score, mood_value)
            
            if mood_score > 0:
                # Normalize rating to [-1, 1] from [1, 5]
                normalized_rating = (rating - 3) / 2
                feature_correlations['mood'].append((mood_score, normalized_rating))
            
            # Check activity correlations
            activity_score = 0
            for activity in self.recommender.activity_keywords:
                if activity in video_data:
                    activity_value = video_data.get(activity, 0)
                    if activity_value > 0:
                        activity_score = max(activity_score, activity_value)
            
            if activity_score > 0:
                normalized_rating = (rating - 3) / 2
                feature_correlations['activity'].append((activity_score, normalized_rating))
            
            # Check genre correlation
            genre = video_data.get('genre')
            if genre:
                # Use preference score as a proxy for genre strength
                genre_score = self.user_preferences.get(user_id, {}).get('favorite_genres', {}).get(genre, 0.5)
                normalized_rating = (rating - 3) / 2
                feature_correlations['genre'].append((genre_score, normalized_rating))
            
            # Check sentiment correlation
            sentiment = video_data.get('sentiment', 0)
            normalized_rating = (rating - 3) / 2
            feature_correlations['sentiment'].append((abs(sentiment), normalized_rating))
            
            # Check popularity correlation
            views = video_data.get('views', 0)
            if views > 0:
                # Normalize views on a log scale
                log_views = min(1.0, np.log1p(views) / 20)  # Cap at 1.0
                normalized_rating = (rating - 3) / 2
                feature_correlations['popularity'].append((log_views, normalized_rating))
        
        # Calculate correlation coefficients
        for feature, data in feature_correlations.items():
            if len(data) < 3:
                continue
                
            # Unzip the data
            feature_values, ratings = zip(*data)
            
            # Calculate Pearson correlation
            try:
                correlation = np.corrcoef(feature_values, ratings)[0, 1]
                
                # Adjust weight based on correlation
                if not np.isnan(correlation):
                    # Scale adjustment by correlation strength
                    adjustment = correlation * 0.1  # Small adjustment per batch
                    
                    # Update the weight
                    self.feature_weights[feature] = max(0.1, min(2.0, self.feature_weights[feature] + adjustment))
            except:
                # Skip if correlation calculation fails
                pass

    def _save_user_data(self, user_id, user_name=None):
        """
        Save user data to disk
        
        Parameters:
        - user_id: User ID
        - user_name: Optional user name
        """
        user_file = os.path.join(self.user_data_dir, f'user_{user_id}.json')
        
        # Prepare data for serialization
        user_data = {
            'user_id': user_id,
            'user_name': user_name,
            'last_updated': datetime.now().isoformat(),
            'history': self.user_history.get(user_id, []),
            'preferences': self.user_preferences.get(user_id, {}),
            'feedback': self.feedback_history.get(user_id, [])
        }
        
        # Remove non-serializable data
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if k != 'embedding'}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return obj
                
        clean_data = clean_for_json(user_data)
        
        # Save to file
        try:
            with open(user_file, 'w') as f:
                json.dump(clean_data, f, indent=2)
        except Exception as e:
            print(f"Error saving user data: {e}")

    def get_user_stats(self, user_id):
        """
        Get statistics about the user's listening history and preferences
        
        Parameters:
        - user_id: User ID
        
        Returns:
        - stats: User statistics
        """
        if user_id not in self.user_history:
            raise ValueError(f"User {user_id} not found")
            
        feedback = self.feedback_history.get(user_id, [])
        preferences = self.user_preferences.get(user_id, {})
        
        # Count sessions
        session_count = len(self.user_history[user_id])
        
        # Count recommendations and feedback
        rec_count = 0
        feedback_count = len(feedback)
        
        for session in self.user_history[user_id]:
            rec_count += len(session['data'].get('recommendations', []))
        
        # Calculate average rating
        ratings = [f.get('rating', 0) for f in feedback if 'rating' in f]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Get top genres
        favorite_genres = preferences.get('favorite_genres', {})
        top_genres = sorted(favorite_genres.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get top artists
        favorite_artists = preferences.get('favorite_artists', {})
        top_artists = sorted(favorite_artists.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get top moods
        favorite_moods = preferences.get('favorite_moods', {})
        top_moods = sorted(favorite_moods.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get top activities
        favorite_activities = preferences.get('favorite_activities', {})
        top_activities = sorted(favorite_activities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Count high-rated songs (4-5 stars)
        high_rated_count = sum(1 for f in feedback if f.get('rating', 0) >= 4)
        
        # Compile stats
        stats = {
            'user_id': user_id,
            'session_count': session_count,
            'recommendation_count': rec_count,
            'feedback_count': feedback_count,
            'average_rating': avg_rating,
            'high_rated_count': high_rated_count,
            'top_genres': top_genres,
            'top_artists': top_artists,
            'top_moods': top_moods,
            'top_activities': top_activities,
            'feature_weights': self.feature_weights.copy()
        }
        
        return stats


class UserInputForm:
    """
    Helper class to generate a structured user input form
    for collecting musical preferences and context
    """
    
    @staticmethod
    def generate_form():
        """
        Generate a structured form for user input
        
        Returns:
        - form: Dictionary containing form fields
        """
        form = {
            'sections': [
                {
                    'title': 'How are you feeling today?',
                    'description': 'Tell us about your current mood and what kind of music you\'re in the mood for.',
                    'fields': [
                        {
                            'id': 'mood_text',
                            'type': 'text_area',
                            'label': 'Describe your mood and music preferences',
                            'placeholder': 'Example: I\'m feeling energetic and want some upbeat music for my workout...',
                            'required': True,
                            'min_length': 20,
                            'max_length': 1000
                        }
                    ]
                },
                {
                    'title': 'Music Preferences',
                    'fields': [
                        {
                            'id': 'favorite_genres',
                            'type': 'multi_select',
                            'label': 'Select up to 3 genres you enjoy',
                            'options': [
                                'Pop', 'Rock', 'Hip-Hop', 'R&B', 'Country', 
                                'Electronic', 'Jazz', 'Classical', 'Folk',
                                'Indie', 'Metal', 'Punk', 'Blues', 'Latin',
                                'Reggae', 'World', 'K-Pop', 'Alternative'
                            ],
                            'required': False,
                            'max_selections': 3,
                            'help_text': 'You can select up to 3 genres'
                        },
                        {
                            'id': 'favorite_artists',
                            'type': 'text',
                            'label': 'Who are some of your favorite artists?',
                            'placeholder': 'E.g., Taylor Swift, The Weeknd, Billie Eilish',
                            'required': False
                        },
                        {
                            'id': 'disliked_genres',
                            'type': 'multi_select',
                            'label': 'Any genres you want to avoid?',
                            'options': [
                                'Pop', 'Rock', 'Hip-Hop', 'R&B', 'Country', 
                                'Electronic', 'Jazz', 'Classical', 'Folk',
                                'Indie', 'Metal', 'Punk', 'Blues', 'Latin',
                                'Reggae', 'World', 'K-Pop', 'Alternative'
                            ],
                            'required': False,
                            'max_selections': 3
                        }
                    ]
                },
                {
                    'title': 'Context',
                    'fields': [
                        {
                            'id': 'activity',
                            'type': 'select',
                            'label': 'What will you be doing while listening?',
                            'options': [
                                'Relaxing', 'Working', 'Studying', 'Exercising',
                                'Partying', 'Driving', 'Cooking', 'Sleeping',
                                'Meditating', 'Gaming', 'Other'
                            ],
                            'required': False
                        },
                        {
                            'id': 'listening_environment',
                            'type': 'select',
                            'label': 'Where will you be listening?',
                            'options': [
                                'Home', 'Work', 'Gym', 'Outdoors', 'In transit',
                                'Party', 'Cafe', 'Other'
                            ],
                            'required': False
                        },
                        {
                            'id': 'time_of_day',
                            'type': 'select',
                            'label': 'Time of day?',
                            'options': [
                                'Morning', 'Afternoon', 'Evening', 'Late night'
                            ],
                            'required': False
                        }
                    ]
                },
                {
                    'title': 'Recommendation Settings',
                    'fields': [
                        {
                            'id': 'diversity',
                            'type': 'slider',
                            'label': 'How diverse do you want your recommendations?',
                            'min': 0,
                            'max': 10,
                            'default': 5,
                            'help_text': 'Lower = more similar songs, Higher = more variety'
                        },
                        {
                            'id': 'playlist_size',
                            'type': 'select',
                            'label': 'How many songs would you like?',
                            'options': [
                                '5', '10', '15', '20', '25'
                            ],
                            'default': '10',
                            'required': True
                        }
                    ]
                }
            ]
        }
        
        return form
    
    @staticmethod
    def process_form_input(form_data):
        """
        Process form input into a format suitable for the recommendation system
        
        Parameters:
        - form_data: Dictionary containing form field values
        
        Returns:
        - processed_data: Processed data for recommendation
        """
        # Extract main text input
        text_input = form_data.get('mood_text', '')
        
        # Build context information
        context = {}
        
        # Add genre preferences
        if 'favorite_genres' in form_data:
            context['genres'] = form_data['favorite_genres']
            
        # Add disliked genres
        if 'disliked_genres' in form_data:
            context['disliked_genres'] = form_data['disliked_genres']
            
        # Add favorite artists
        if 'favorite_artists' in form_data:
            # Split comma-separated list
            artists = [a.strip() for a in form_data['favorite_artists'].split(',')]
            context['artists'] = artists
            
        # Add activity
        if 'activity' in form_data:
            context['activity'] = form_data['activity']
            
        # Add environment
        if 'listening_environment' in form_data:
            context['environment'] = form_data['listening_environment']
            
        # Add time of day
        if 'time_of_day' in form_data:
            context['time_of_day'] = form_data['time_of_day']
            
        # Add diversity setting
        if 'diversity' in form_data:
            try:
                diversity = int(form_data['diversity'])
                context['diversity'] = diversity / 10.0  # Convert 0-10 to 0-1 scale
            except:
                context['diversity'] = 0.5  # Default
                
        # Add playlist size
        if 'playlist_size' in form_data:
            try:
                context['playlist_size'] = int(form_data['playlist_size'])
            except:
                context['playlist_size'] = 10  # Default
        
        # Enhance text input with context if it's too short
        if len(text_input) < 50 and context:
            # Build additional context description
            context_text = []
            
            if 'genres' in context and context['genres']:
                genre_text = ', '.join(context['genres'][:3])
                context_text.append(f"I enjoy {genre_text} music")
                
            if 'artists' in context and context['artists']:
                artist_text = ', '.join(context['artists'][:3])
                context_text.append(f"I like artists such as {artist_text}")
                
            if 'activity' in context:
                context_text.append(f"I'll be {context['activity'].lower()}")
                
            if 'environment' in context:
                context_text.append(f"I'm at {context['environment'].lower()}")
                
            if 'time_of_day' in context:
                context_text.append(f"It's {context['time_of_day'].lower()} now")
                
            # Add to text input
            if context_text:
                additional = '. '.join(context_text)
                text_input = f"{text_input}. Also, {additional}."
        
        return {
            'text_input': text_input,
            'context': context
        }


class FeedbackForm:
    """
    Helper class to generate and process feedback forms
    for collecting user ratings and preferences
    """
    
    @staticmethod
    def generate_feedback_form(video_data):
        """
        Generate a feedback form for a specific video
        
        Parameters:
        - video_data: Video data from recommendation
        
        Returns:
        - form: Feedback form structure
        """
        form = {
            'video_id': video_data['video_id'],
            'title': video_data['title'],
            'artist': video_data['artist'],
            'fields': [
                {
                    'id': 'rating',
                    'type': 'rating',
                    'label': 'How would you rate this song?',
                    'min': 1,
                    'max': 5,
                    'required': True
                },
                {
                    'id': 'listen_duration',
                    'type': 'hidden',
                    'value': 0  # Will be updated by frontend
                },
                {
                    'id': 'skip_reason',
                    'type': 'select',
                    'label': 'If you skipped this song, why?',
                    'options': [
                        'Not applicable', 'Don\'t like the genre', 'Don\'t like the artist',
                        'Not in the mood', 'Too slow', 'Too fast', 'Too loud',
                        'Too quiet', 'Heard it too many times', 'Other'
                    ],
                    'required': False,
                    'default': 'Not applicable'
                },
                {
                    'id': 'comments',
                    'type': 'text_area',
                    'label': 'Any comments about this song? (Optional)',
                    'required': False,
                    'placeholder': 'What did you like or dislike about it?'
                }
            ]
        }
        
        return form
    
    @staticmethod
    def process_feedback(form_data, session_id):
        """
        Process a feedback form submission
        
        Parameters:
        - form_data: Submitted form data
        - session_id: Current session ID
        
        Returns:
        - feedback_data: Processed feedback data
        """
        # Convert rating to integer before comparison
        rating = int(form_data.get('rating', 3))
        
        feedback = {
            'video_id': form_data.get('video_id'),
            'session_id': session_id,
            'rating': rating,
            'listen_duration': float(form_data.get('listen_duration', 0)),
            'skip_reason': form_data.get('skip_reason') if rating <= 2 else None,
            'comments': form_data.get('comments', '')
        }
        
        return feedback


class XGBoostRecommender:
    def __init__(self, base_recommender):
        """
        Initialize XGBoost-based recommender
        
        Parameters:
        - base_recommender: An instance of FeedbackEnhancedRecommender
        """
        self.recommender = base_recommender
        self.model = None
        self.label_encoders = {}
        self.feature_columns = [
            'genre', 'mood', 'activity', 'sentiment', 'views',
            'likes', 'comment_count', 'duration'
        ]
        self.min_feedback_threshold = 3  # Raised to 3 feedback points
        
    def train_model(self, user_id):
        """
        Train XGBoost model on user's feedback history
        
        Parameters:
        - user_id: User ID to train model for
        """
        print("\n=== XGBoost Training Debug ===")
        print(f"Attempting to train XGBoost model for user {user_id}")
        
        # Get user's feedback history
        feedback = self.recommender.feedback_history.get(user_id, [])
        print(f"Found {len(feedback)} feedback points for user")
        
        if len(feedback) < self.min_feedback_threshold:
            print(f"Not enough feedback to train XGBoost model. Need {self.min_feedback_threshold} feedback points, have {len(feedback)}.")
            return False
            
        print(f"Training XGBoost model with {len(feedback)} feedback points...")
            
        # Prepare training data
        X, y = self._prepare_training_data(feedback)
        print(f"Prepared training data shape: X={X.shape}, y={y.shape}")
        
        if len(X) < self.min_feedback_threshold:
            print(f"Not enough valid samples to train XGBoost model. Need {self.min_feedback_threshold} samples, have {len(X)}.")
            return False
            
        # Handle single sample case
        if len(X) == 1:
            print("Single sample detected - using all data for training without splitting")
            X_train, y_train = X, y
            X_test, y_test = X, y  # Use same data for validation
        else:
            # Split data for multiple samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 4, 5],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'n_estimators': [100, 200],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.1, 1.0, 5.0]
        }
        
        # Adjust grid size based on dataset size
        n_samples = len(X_train)
        if n_samples < 10:
            # Use smaller grid for small datasets
            param_grid = {
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 4],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'n_estimators': [100],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.1, 1.0]
            }
        
        print("Performing GridSearchCV with parameters:")
        print(param_grid)
        
        # Initialize base model with early stopping
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            early_stopping_rounds=10
        )
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=min(3, n_samples),  # Use 3-fold CV or less if not enough samples
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        
        # Fit GridSearchCV
        print("Starting GridSearchCV...")
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Get best parameters
        best_params = grid_search.best_params_
        print("Best parameters found:", best_params)
        print("Best cross-validation score:", -grid_search.best_score_)  # Convert back to positive MSE
        
        # Train final model with best parameters
        self.model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            random_state=42,
            early_stopping_rounds=10
        )
        
        # Train the model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate and print performance metrics
        y_train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test set predictions and metrics
        y_test_pred = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\n=== Model Performance Metrics ===")
        print("\nTraining Set Performance:")
        print(f"Mean Squared Error (MSE): {train_mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {train_mae:.4f}")
        print(f"R-squared (R²): {train_r2:.4f}")
        
        print("\nTest Set Performance:")
        print(f"Mean Squared Error (MSE): {test_mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
        print(f"R-squared (R²): {test_r2:.4f}")
        
        # Print sample of actual vs predicted values
        print("\n=== Sample of Actual vs Predicted Ratings ===")
        print("\nTraining Set (first 5 samples):")
        print("Actual Rating | Predicted Rating | Difference")
        print("-" * 45)
        for actual, pred in zip(y_train[:5], y_train_pred[:5]):
            print(f"{actual:12.1f} | {pred:14.2f} | {abs(actual - pred):10.2f}")
            
        print("\nTest Set (first 5 samples):")
        print("Actual Rating | Predicted Rating | Difference")
        print("-" * 45)
        for actual, pred in zip(y_test[:5], y_test_pred[:5]):
            print(f"{actual:12.1f} | {pred:14.2f} | {abs(actual - pred):10.2f}")
        
        # Calculate feature importance
        feature_importance = self.model.feature_importances_
        feature_names = self.feature_columns
        
        # Create feature importance dictionary
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        print("\nFeature Importance:")
        for feature, importance in sorted_importance.items():
            print(f"{feature}: {importance:.4f}")
        
        print("\nXGBoost model trained successfully!")
        return True
        
    def _prepare_training_data(self, feedback):
        """
        Prepare training data from feedback history with proper feature encoding and scaling
        """
        print("\n=== Preparing Training Data ===")
        X = []
        y = []
        feature_data = {col: [] for col in self.feature_columns}
        
        # First pass: collect all data for proper encoding
        for item in feedback:
            video_data = item.get('video_data', {})
            if not video_data:
                continue
                
            # Collect features
            for col in self.feature_columns:
                value = video_data.get(col)
                feature_data[col].append(value)
            
            y.append(item.get('rating', 0))
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.scalers = {}
        
        # Process each feature
        processed_features = []
        for col in self.feature_columns:
            values = feature_data[col]
            
            # Handle categorical features
            if col in ['genre', 'mood', 'activity']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                # Fit and transform categorical data
                encoded_values = self.label_encoders[col].fit_transform([v for v in values if v is not None])
                # Handle None values by using -1
                processed_values = []
                for i, v in enumerate(values):
                    if v is None:
                        processed_values.append(-1)
                    else:
                        processed_values.append(encoded_values[i])
                processed_features.append(processed_values)
                
            # Handle numerical features
            elif col in ['views', 'likes', 'comment_count', 'duration']:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                # Convert to float and handle None values
                numeric_values = [float(v) if v is not None else 0.0 for v in values]
                # Scale numerical data
                scaled_values = self.scalers[col].fit_transform(np.array(numeric_values).reshape(-1, 1)).flatten()
                processed_features.append(scaled_values)
                
            # Handle sentiment
            elif col == 'sentiment':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                # Convert to float and handle None values
                sentiment_values = [float(v) if v is not None else 0.0 for v in values]
                # Scale sentiment data
                scaled_values = self.scalers[col].fit_transform(np.array(sentiment_values).reshape(-1, 1)).flatten()
                processed_features.append(scaled_values)
        
        # Transpose the processed features to get samples
        X = np.array(processed_features).T
        
        # Print feature statistics
        print("\nFeature Statistics:")
        for i, col in enumerate(self.feature_columns):
            print(f"\n{col}:")
            print(f"  Mean: {np.mean(X[:, i]):.4f}")
            print(f"  Std: {np.std(X[:, i]):.4f}")
            print(f"  Min: {np.min(X[:, i]):.4f}")
            print(f"  Max: {np.max(X[:, i]):.4f}")
            print(f"  Unique values: {len(np.unique(X[:, i]))}")
        
        # Print target statistics
        y = np.array(y)
        print("\nTarget (Rating) Statistics:")
        print(f"  Mean: {np.mean(y):.4f}")
        print(f"  Std: {np.std(y):.4f}")
        print(f"  Min: {np.min(y):.4f}")
        print(f"  Max: {np.max(y):.4f}")
        print(f"  Unique values: {len(np.unique(y))}")
        
        return X, y
        
    def predict_ratings(self, videos):
        """
        Predict ratings for a list of videos
        
        Parameters:
        - videos: List of video data dictionaries
        
        Returns:
        - List of predicted ratings
        """
        if not self.model:
            print("Using default ratings (0.5) as XGBoost model is not trained")
            return [0.5] * len(videos)  # Default rating if no model
            
        print("Using XGBoost model to predict ratings...")
        # Prepare features
        X = []
        for video in videos:
            features = []
            for col in self.feature_columns:
                value = video.get(col)
                
                # Handle categorical features
                if col in ['genre', 'mood', 'activity']:
                    if col in self.label_encoders and value:
                        try:
                            # Transform the value using the fitted encoder
                            value = self.label_encoders[col].transform([value])[0]
                        except ValueError:
                            # If value not seen during training, use -1
                            value = -1
                    else:
                        value = -1
                
                # Handle numerical features
                elif col in ['views', 'likes', 'comment_count', 'duration']:
                    value = float(value) if value else 0
                    
                # Handle sentiment
                elif col == 'sentiment':
                    value = float(value) if value else 0
                    
                features.append(value)
            
            # Ensure we have all features
            if len(features) != len(self.feature_columns):
                print(f"Warning: Missing features for video {video.get('video_id')}. Expected {len(self.feature_columns)}, got {len(features)}")
                # Pad with zeros if missing features
                features.extend([0] * (len(self.feature_columns) - len(features)))
            
            X.append(features)
            
        # Convert to numpy array and ensure correct shape
        X = np.array(X)
        print(f"Prepared features shape: {X.shape}")
        
        # Make predictions
        predictions = self.model.predict(X)
        return predictions.tolist()

# Example usage

def run_example():
    """Run a complete example of the feedback-enhanced recommendation system"""
    # Initialize the base recommender
    from enhanced_recommender import EnhancedMusicRecommender
    
    base_recommender = EnhancedMusicRecommender()
    base_recommender.load_data()
    base_recommender.preprocess_comments()
    base_recommender.create_embeddings()
    base_recommender.cluster_videos()
    
    # Initialize the feedback-enhanced recommender
    recommender = FeedbackEnhancedRecommender(base_recommender)
    
    # Create a new user
    user_id = recommender.get_or_create_user(user_name="TestUser")
    print(f"Created user with ID: {user_id}")
    
    # Generate a form for user input
    input_form = UserInputForm.generate_form()
    print("Generated user input form with sections:", [s['title'] for s in input_form['sections']])
    
    # Simulate user input
    form_data = {
        'mood_text': "I'm feeling energetic today and want to go for a run. I need some upbeat music to keep me motivated.",
        'favorite_genres': ['Pop', 'Rock', 'Electronic'],
        'activity': 'Exercising',
        'listening_environment': 'Outdoors',
        'time_of_day': 'Morning',
        'diversity': 7,
        'playlist_size': 5
    }
    
    # Process the form input
    processed_input = UserInputForm.process_form_input(form_data)
    print("Processed user input:", processed_input['text_input'])
    
    # Create a new session
    session_id = recommender.process_user_input(
        user_id, 
        processed_input['text_input'],
        processed_input['context']
    )
    print(f"Created session with ID: {session_id}")
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(
        session_id=session_id,
        batch_size=int(processed_input['context'].get('playlist_size', 5)),
        strategy='diverse'
    )
    
    print(f"Generated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} by {rec['artist']} ({rec['recommendation_method']})")
    
    # Generate feedback forms for each recommendation
    feedback_forms = [FeedbackForm.generate_feedback_form(rec) for rec in recommendations]
    print(f"Generated {len(feedback_forms)} feedback forms")
    
    # Simulate user feedback (random ratings)
    import random
    
    for i, rec in enumerate(recommendations):
        # Simulate some realistic feedback (e.g., user likes 3 out of 5 songs)
        if i % 5 < 3:  # 60% are liked
            rating = random.randint(4, 5)
            listen_duration = random.uniform(30, 180)  # 30s to 3min
            skip_reason = "Not applicable"
        else:  # 40% are disliked or neutral
            rating = random.randint(1, 3)
            listen_duration = random.uniform(5, 60)  # 5s to 1min
            skip_reasons = ["Don't like the genre", "Not in the mood", "Too slow", "Heard it too many times"]
            skip_reason = random.choice(skip_reasons)
        
        # Create feedback data
        form_data = {
            'video_id': rec['video_id'],
            'rating': rating,
            'listen_duration': listen_duration,
            'skip_reason': skip_reason,
            'comments': ""
        }
        
        # Process feedback
        feedback_data = FeedbackForm.process_feedback(form_data, session_id)
        
        # Record feedback
        recommender.record_feedback(
            session_id=session_id,
            video_id=rec['video_id'],
            rating=feedback_data['rating'],
            skip_reason=feedback_data['skip_reason'],
            listen_duration=feedback_data['listen_duration']
        )
        
        print(f"Recorded feedback for {rec['title']}: {rating}/5 stars")
    
    # Simulate a second session with slightly different mood
    print("\n=== Second session ===")
    
    form_data2 = {
        'mood_text': "I'm winding down after work and want something relaxing. Looking for chill music to help me relax.",
        'favorite_genres': ['R&B', 'Jazz', 'Indie'],
        'activity': 'Relaxing',
        'listening_environment': 'Home',
        'time_of_day': 'Evening',
        'diversity': 5,
        'playlist_size': 5
    }
    
    processed_input2 = UserInputForm.process_form_input(form_data2)
    session_id2 = recommender.process_user_input(
        user_id, 
        processed_input2['text_input'],
        processed_input2['context']
    )
    
    recommendations2 = recommender.generate_recommendations(
        session_id=session_id2,
        batch_size=int(processed_input2['context'].get('playlist_size', 5)),
        strategy='focused'
    )
    
    print(f"Generated {len(recommendations2)} recommendations:")
    for i, rec in enumerate(recommendations2, 1):
        print(f"{i}. {rec['title']} by {rec['artist']} ({rec['recommendation_method']})")
    
    # Simulate feedback
    for i, rec in enumerate(recommendations2):
        rating = random.randint(1, 5)
        listen_duration = random.uniform(10, 200)
        skip_reason = "Not applicable" if rating > 2 else random.choice(
            ["Not in the mood", "Too slow", "Too fast"]
        )
        
        recommender.record_feedback(
            session_id=session_id2,
            video_id=rec['video_id'],
            rating=rating,
            skip_reason=skip_reason,
            listen_duration=listen_duration
        )
    
    # Generate a final personalized playlist
    print("\n=== Final Personalized Playlist ===")
    
    final_playlist = recommender.generate_final_playlist(
        user_id=user_id,
        name="Your Perfect Mix",
        size=10
    )
    
    print(f"Generated final playlist: {final_playlist['name']}")
    print(f"Description: {final_playlist['description']}")
    print(f"Contains {final_playlist['size']} tracks")
    
    for i, track in enumerate(final_playlist['tracks'], 1):
        if 'user_rating' in track:
            print(f"{i}. {track['title']} by {track['artist']} (Your rating: {track['user_rating']}/5)")
        else:
            print(f"{i}. {track['title']} by {track['artist']} (Recommended based on your preferences)")
    
    # Get user stats
    user_stats = recommender.get_user_stats(user_id)
    
    print("\n=== User Stats ===")
    print(f"Sessions: {user_stats['session_count']}")
    print(f"Recommendations received: {user_stats['recommendation_count']}")
    print(f"Feedback provided: {user_stats['feedback_count']}")
    print(f"Average rating: {user_stats['average_rating']:.2f}/5")
    
    if user_stats['top_genres']:
        print("\nTop Genres:")
        for genre, score in user_stats['top_genres']:
            print(f"- {genre}: {score:.2f}")
    
    if user_stats['top_artists']:
        print("\nTop Artists:")
        for artist, score in user_stats['top_artists']:
            print(f"- {artist}: {score:.2f}")
    
    if user_stats['top_moods']:
        print("\nPreferred Moods:")
        for mood, score in user_stats['top_moods']:
            print(f"- {mood}: {score:.2f}")
    
    print("\nFeature Weights:")
    for feature, weight in user_stats['feature_weights'].items():
        print(f"- {feature}: {weight:.2f}")
    
    # Clean up
    base_recommender.close()

if __name__ == "__main__":
    run_example()