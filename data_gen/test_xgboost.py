from feedback_recommender import FeedbackEnhancedRecommender
from enhanced_recommender import EnhancedMusicRecommender
import random

def test_xgboost_recommender():
    print("\n=== Starting XGBoost Recommender Test ===")
    
    # Initialize base recommender
    base_recommender = EnhancedMusicRecommender()
    base_recommender.load_data()
    base_recommender.preprocess_comments()
    base_recommender.create_embeddings()
    base_recommender.cluster_videos()
    
    # Initialize feedback recommender
    recommender = FeedbackEnhancedRecommender(base_recommender)
    
    # Create test user
    user_id = recommender.get_or_create_user(user_name="TestUser")
    print(f"\nCreated test user with ID: {user_id}")
    
    # Create test session
    session_id = recommender.process_user_input(
        user_id,
        "I'm feeling energetic and want some upbeat music for my workout",
        {
            'activity': 'Exercising',
            'genres': ['Pop', 'Rock', 'Electronic']
        }
    )
    print(f"\nCreated test session with ID: {session_id}")
    
    # Get initial recommendations
    recommendations = recommender.generate_recommendations(
        session_id=session_id,
        batch_size=10,
        strategy='diverse'
    )
    print(f"\nGenerated {len(recommendations)} initial recommendations")
    
    # Test genres and moods
    test_genres = ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'R&B']
    test_moods = ['energetic', 'happy', 'relaxed', 'sad', 'angry']
    
    # Record feedback for each recommendation
    feedback_data = []
    for i, rec in enumerate(recommendations):
        # Simulate different feedback patterns
        if i < 3:  # First 3 songs: High ratings
            rating = random.uniform(4.0, 5.0)
            genres = ['Pop', 'Rock']
            moods = ['energetic', 'happy']
        elif i < 7:  # Next 4 songs: Medium ratings
            rating = random.uniform(3.0, 4.0)
            genres = random.sample(test_genres, 2)
            moods = random.sample(test_moods, 2)
        else:  # Last 3 songs: Low ratings
            rating = random.uniform(1.0, 2.0)
            genres = ['Hip-Hop', 'R&B']
            moods = ['sad', 'angry']
        
        # Simulate listen duration (30-300 seconds)
        listen_duration = random.uniform(30, 300)
        
        # Create feedback data
        feedback = {
            'video_id': rec['video_id'],
            'rating': rating,
            'listen_duration': listen_duration,
            'genres': genres,
            'moods': moods
        }
        feedback_data.append(feedback)
        
        print(f"\nRecording feedback for video {rec['video_id']}:")
        print(f"Rating: {rating:.1f}")
        print(f"Listen Duration: {listen_duration:.1f}s")
        print(f"Genres: {genres}")
        print(f"Moods: {moods}")
        
        # Record feedback
        recommender.record_feedback(
            session_id=session_id,
            video_id=rec['video_id'],
            rating=rating,
            listen_duration=listen_duration
        )
    
    # Train the model
    print("\nTraining XGBoost model...")
    recommender.xgboost_recommender.train_model(user_id)
    
    # Get new recommendations after training
    new_recommendations = recommender.generate_recommendations(
        session_id=session_id,
        batch_size=5,
        strategy='focused'
    )
    print(f"\nGenerated {len(new_recommendations)} new recommendations after training")
    
    # Get user statistics
    user_stats = recommender.get_user_statistics(user_id)
    print("\nUser Statistics:")
    print(f"Total Feedback Points: {user_stats['total_feedback']}")
    print(f"Average Rating: {user_stats['average_rating']:.2f}")
    print(f"Top Genres: {user_stats['top_genres']}")
    
    # Print feedback summary
    print("\nFeedback Summary:")
    print(f"Total feedback points: {len(feedback_data)}")
    print("\nRating Distribution:")
    rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for feedback in feedback_data:
        rating = round(feedback['rating'])
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    for rating, count in sorted(rating_counts.items()):
        print(f"{rating} stars: {count} songs")
    
    print("\nGenre Distribution:")
    genre_counts = {}
    for feedback in feedback_data:
        for genre in feedback['genres']:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    for genre, count in sorted(genre_counts.items()):
        print(f"{genre}: {count} songs")
    
    print("\nMood Distribution:")
    mood_counts = {}
    for feedback in feedback_data:
        for mood in feedback['moods']:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
    for mood, count in sorted(mood_counts.items()):
        print(f"{mood}: {count} songs")
    
    # Print XGBoost model status
    print("\nXGBoost Model Status:")
    print(f"Model Trained: {recommender.xgboost_recommender.is_trained}")
    print(f"Model Active: {recommender.xgboost_recommender.is_active}")
    
    if recommender.xgboost_recommender.is_trained:
        print("\nFeature Importance:")
        for feature, importance in recommender.xgboost_recommender.feature_importance.items():
            print(f"{feature}: {importance:.4f}")
    
    # Close the base recommender
    base_recommender.close()
    
    print("\n=== XGBoost Recommender Test Completed ===")

if __name__ == "__main__":
    test_xgboost_recommender() 