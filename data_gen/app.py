from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import uuid
from datetime import datetime

from enhanced_recommender import EnhancedMusicRecommender
from feedback_recommender import FeedbackEnhancedRecommender, UserInputForm, FeedbackForm

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize the recommenders
base_recommender = None
feedback_recommender = None

# Loading state
loading_state = {
    'is_loading': True,
    'status': 'Initializing...',
    'progress': 0
}

def init_recommenders():
    """Initialize the recommender systems in the background"""
    global base_recommender, feedback_recommender, loading_state
    
    try:
        loading_state = {
            'is_loading': True,
            'status': 'Loading recommender model...',
            'progress': 10
        }
        
        # Initialize base recommender
        base_recommender = EnhancedMusicRecommender(cache_dir='./cache')
        
        loading_state['status'] = 'Loading data from database...'
        loading_state['progress'] = 20
        base_recommender.load_data()
        
        loading_state['status'] = 'Processing comments...'
        loading_state['progress'] = 40
        base_recommender.preprocess_comments()
        
        loading_state['status'] = 'Creating embeddings...'
        loading_state['progress'] = 60
        base_recommender.create_embeddings()
        
        loading_state['status'] = 'Clustering videos...'
        loading_state['progress'] = 80
        base_recommender.cluster_videos()
        
        # Initialize feedback-enhanced recommender
        loading_state['status'] = 'Initializing feedback system...'
        loading_state['progress'] = 90
        feedback_recommender = FeedbackEnhancedRecommender(base_recommender)
        
        loading_state['status'] = 'Ready'
        loading_state['progress'] = 100
        loading_state['is_loading'] = False
        
        print("Recommender systems initialized successfully")
    except Exception as e:
        loading_state['status'] = f"Error: {str(e)}"
        loading_state['is_loading'] = False
        print(f"Error initializing recommender systems: {e}")

# Start initialization in a separate thread
import threading
init_thread = threading.Thread(target=init_recommenders)
init_thread.daemon = True
init_thread.start()

# Helper function to ensure recommenders are ready
def ensure_recommenders_ready():
    """Ensure recommenders are initialized"""
    if feedback_recommender is None or base_recommender is None:
        return False
    return not loading_state['is_loading']

# Routes
@app.route('/')
def index():
    """Home page"""
    # Check if user is logged in
    user_id = session.get('user_id')
    
    return render_template('index.html', 
                           loading=loading_state,
                           user_id=user_id)

@app.route('/api/status')
def status():
    """Get loading status"""
    return jsonify(loading_state)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        # Handle login or new user creation
        action = request.form.get('action')
        
        if action == 'new_user':
            # Create a new user
            user_name = request.form.get('user_name', '')
            
            if not ensure_recommenders_ready():
                return render_template('login.html', 
                                    error="System is still initializing. Please wait a moment.")
            
            # Create new user
            user_id = feedback_recommender.get_or_create_user(user_name=user_name)
            
            # Store in session
            session['user_id'] = user_id
            session['user_name'] = user_name
            
            return redirect(url_for('dashboard'))
            
        elif action == 'login':
            # Login existing user
            user_id = request.form.get('user_id')
            
            if not user_id:
                return render_template('login.html', 
                                    error="Please provide a user ID")
            
            if not ensure_recommenders_ready():
                return render_template('login.html', 
                                    error="System is still initializing. Please wait a moment.")
            
            try:
                # Verify user exists
                user_id = feedback_recommender.get_or_create_user(
                    user_id=user_id, 
                    create_if_not_exists=False
                )
                
                # Store in session
                session['user_id'] = user_id
                
                return redirect(url_for('dashboard'))
            except ValueError:
                return render_template('login.html', 
                                    error="User ID not found")
    
    # GET request
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Log out the user"""
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    if not ensure_recommenders_ready():
        return render_template('loading.html', 
                            loading=loading_state,
                            redirect_url=url_for('dashboard'))
    
    # Get user stats
    try:
        user_stats = feedback_recommender.get_user_stats(user_id)
    except Exception as e:
        user_stats = {
            'session_count': 0,
            'recommendation_count': 0,
            'feedback_count': 0,
            'average_rating': 0,
            'top_genres': [],
            'top_artists': [],
            'top_moods': [],
            'high_rated_count': 0
        }
    
    return render_template('dashboard.html', 
                          user_id=user_id,
                          user_name=session.get('user_name', 'User'),
                          stats=user_stats)

@app.route('/new_recommendations', methods=['GET', 'POST'])
def new_recommendations():
    """Create new recommendations"""
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    if not ensure_recommenders_ready():
        return render_template('loading.html', 
                            loading=loading_state,
                            redirect_url=url_for('new_recommendations'))
    
    if request.method == 'POST':
        # Process form submission
        form_data = request.form.to_dict()
        
        # Process the input
        processed_input = UserInputForm.process_form_input(form_data)
        
        # Create a new session
        session_id = feedback_recommender.process_user_input(
            user_id, 
            processed_input['text_input'],
            processed_input['context']
        )
        
        # Store session_id
        session['current_session_id'] = session_id
        
        # Redirect to recommendations page
        return redirect(url_for('view_recommendations'))
    
    # Generate the input form
    input_form = UserInputForm.generate_form()
    
    return render_template('new_recommendations.html', 
                          user_id=user_id,
                          form=input_form)

@app.route('/recommendations')
def view_recommendations():
    """View current recommendations"""
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Get current session
    session_id = session.get('current_session_id')
    if not session_id:
        return redirect(url_for('new_recommendations'))
    
    if not ensure_recommenders_ready():
        return render_template('loading.html', 
                            loading=loading_state,
                            redirect_url=url_for('view_recommendations'))
    
    # Get recommendations
    try:
        # Default to 10 recommendations with diverse strategy
        recommendations = feedback_recommender.generate_recommendations(
            session_id=session_id,
            batch_size=10,
            strategy='diverse'
        )
    except Exception as e:
        # Handle error
        return render_template('error.html', 
                              error=f"Error generating recommendations: {str(e)}")
    
    # Add YouTube watch URLs
    for rec in recommendations:
        rec['youtube_url'] = f"https://www.youtube.com/watch?v={rec['video_id']}"
        rec['embed_url'] = f"https://www.youtube.com/embed/{rec['video_id']}"
    
    return render_template('recommendations.html', 
                          user_id=user_id,
                          session_id=session_id,
                          recommendations=recommendations)

@app.route('/feedback/<video_id>', methods=['GET', 'POST'])
def feedback(video_id):
    """Provide feedback for a recommendation"""
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Get current session
    session_id = session.get('current_session_id')
    if not session_id:
        return redirect(url_for('dashboard'))
    
    if not ensure_recommenders_ready():
        return render_template('loading.html', 
                            loading=loading_state,
                            redirect_url=url_for('feedback', video_id=video_id))
    
    if request.method == 'POST':
        # Process feedback submission
        form_data = request.form.to_dict()
        form_data['video_id'] = video_id
        
        # Process the feedback
        feedback_data = FeedbackForm.process_feedback(form_data, session_id)
        
        # Record feedback
        try:
            feedback_recommender.record_feedback(
                session_id=session_id,
                video_id=video_id,
                rating=feedback_data['rating'],
                skip_reason=feedback_data['skip_reason'],
                listen_duration=feedback_data['listen_duration']
            )
            
            # Redirect back to recommendations
            return redirect(url_for('view_recommendations'))
        except Exception as e:
            return render_template('error.html', 
                                  error=f"Error recording feedback: {str(e)}")
    
    # Find the video in current recommendations
    try:
        for rec in feedback_recommender.session_recommendations.get(session_id, []):
            if rec['video_id'] == video_id:
                video_data = rec
                break
        else:
            return render_template('error.html', 
                                 error=f"Video {video_id} not found in current session")
    except Exception as e:
        return render_template('error.html', 
                             error=f"Error finding video: {str(e)}")
    
    # Generate feedback form
    feedback_form = FeedbackForm.generate_feedback_form(video_data)
    
    # Add YouTube URLs
    video_data['youtube_url'] = f"https://www.youtube.com/watch?v={video_id}"
    video_data['embed_url'] = f"https://www.youtube.com/embed/{video_id}"
    
    return render_template('feedback.html', 
                          user_id=user_id,
                          session_id=session_id,
                          video=video_data,
                          form=feedback_form)

@app.route('/final_playlist')
def final_playlist():
    """Generate and view final personalized playlist"""
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    if not ensure_recommenders_ready():
        return render_template('loading.html', 
                            loading=loading_state,
                            redirect_url=url_for('final_playlist'))
    
    # Generate final playlist
    try:
        playlist = feedback_recommender.generate_final_playlist(
            user_id=user_id,
            size=20
        )
        
        # Add YouTube URLs
        for track in playlist['tracks']:
            track['youtube_url'] = f"https://www.youtube.com/watch?v={track['video_id']}"
            track['embed_url'] = f"https://www.youtube.com/embed/{track['video_id']}"
        
        return render_template('final_playlist.html', 
                              user_id=user_id,
                              playlist=playlist)
    except Exception as e:
        return render_template('error.html', 
                             error=f"Error generating final playlist: {str(e)}")

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """API endpoint for generating recommendations"""
    # Check authentication
    user_id = request.json.get('user_id')
    api_key = request.json.get('api_key')
    
    # Very simple auth check
    if not user_id or api_key != 'dev_api_key':
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ensure_recommenders_ready():
        return jsonify({'error': 'System initializing', 'status': loading_state}), 503
    
    # Get recommendation parameters
    text_input = request.json.get('text', '')
    context = request.json.get('context', {})
    batch_size = int(request.json.get('size', 10))
    strategy = request.json.get('strategy', 'diverse')
    
    try:
        # Create a new session
        session_id = feedback_recommender.process_user_input(
            user_id, 
            text_input,
            context
        )
        
        # Generate recommendations
        recommendations = feedback_recommender.generate_recommendations(
            session_id=session_id,
            batch_size=batch_size,
            strategy=strategy
        )
        
        # Add YouTube URLs
        for rec in recommendations:
            rec['youtube_url'] = f"https://www.youtube.com/watch?v={rec['video_id']}"
        
        return jsonify({
            'session_id': session_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Handle feedback submission"""
    print("\n=== API Feedback Debug ===")
    print("Received feedback request")
    
    # Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        print("No user ID in session")
        return jsonify({'error': 'Not logged in'}), 401
    
    # Get feedback data
    data = request.get_json()
    print(f"Feedback data: {data}")
    
    if not data:
        print("No feedback data received")
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Extract feedback data
        video_id = data.get('video_id')
        rating = int(data.get('rating', 0))
        skip_reason = data.get('skip_reason')
        listen_duration = float(data.get('listen_duration', 0))
        session_id = data.get('session_id')
        
        print(f"Processing feedback for video {video_id}")
        print(f"Rating: {rating}")
        print(f"Session ID: {session_id}")
        
        if not video_id or not rating or not session_id:
            print("Missing required feedback data")
            return jsonify({'error': 'Missing required data'}), 400
        
        # Record feedback
        feedback = feedback_recommender.record_feedback(
            session_id=session_id,
            video_id=video_id,
            rating=rating,
            skip_reason=skip_reason,
            listen_duration=listen_duration
        )
        
        print("Feedback recorded successfully")
        return jsonify({'success': True, 'feedback': feedback})
        
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/final_playlist', methods=['POST'])
def api_final_playlist():
    """API endpoint for generating final playlist"""
    # Check authentication
    user_id = request.json.get('user_id')
    api_key = request.json.get('api_key')
    
    # Very simple auth check
    if not user_id or api_key != 'dev_api_key':
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ensure_recommenders_ready():
        return jsonify({'error': 'System initializing', 'status': loading_state}), 503
    
    # Get parameters
    name = request.json.get('name')
    size = int(request.json.get('size', 20))
    description = request.json.get('description')
    
    try:
        # Generate final playlist
        playlist = feedback_recommender.generate_final_playlist(
            user_id=user_id,
            name=name,
            size=size,
            description=description
        )
        
        # Add YouTube URLs
        for track in playlist['tracks']:
            track['youtube_url'] = f"https://www.youtube.com/watch?v={track['video_id']}"
        
        return jsonify(playlist)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.template_filter('format_date')
def format_date(value):
    """Format ISO date string to readable format"""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return value

# Create HTML templates
def create_templates():
    """Create HTML templates for the application"""
    os.makedirs('templates', exist_ok=True)
    
    # Base template
    with open('templates/base.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}MoodTunes{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { 
            padding-top: 20px; 
            background-color: #f8f9fa;
        }
        .header {
            padding: 20px 0;
            text-align: center;
            margin-bottom: 30px;
            background-color: #343a40;
            color: white;
            border-radius: 10px;
        }
        .card {
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .video-container {
            position: relative;
            padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
            height: 0;
            overflow: hidden;
            border-radius: 8px;
        }
        .video-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        .rating {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .rating input {
            display: none;
        }
        .rating label {
            cursor: pointer;
            font-size: 30px;
            color: #ddd;
            padding: 5px;
        }
        .rating label:hover,
        .rating label:hover ~ label,
        .rating input:checked ~ label {
            color: #f8b739;
        }
        .progress-bar {
            background-color: #6c5ce7;
        }
        .btn-primary {
            background-color: #6c5ce7;
            border-color: #6c5ce7;
        }
        .btn-primary:hover {
            background-color: #5b4cc7;
            border-color: #5b4cc7;
        }
        .playlist-card {
            border-left: 4px solid #6c5ce7;
        }
        .recommendation-method {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
        }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <div class="container">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark rounded mb-4">
            <div class="container-fluid">
                <a class="navbar-brand" href="{{ url_for('index') }}">
                    <i class="fas fa-music me-2"></i>MoodTunes
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                        </li>
                        {% if user_id %}
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('dashboard') }}">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('new_recommendations') }}">New Recommendations</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('final_playlist') }}">My Playlist</a>
                        </li>
                        {% endif %}
                    </ul>
                    <div class="navbar-nav">
                        {% if user_id %}
                        <span class="nav-item nav-link text-light">
                            <i class="fas fa-user me-1"></i> {{ user_id }}
                        </span>
                        <a class="nav-link" href="{{ url_for('logout') }}">Logout</a>
                        {% else %}
                        <a class="nav-link" href="{{ url_for('login') }}">Login</a>
                        {% endif %}
                    </div>
                </div>
            </div>
        </nav>
        
        {% block content %}{% endblock %}
        
        <footer class="mt-5 text-center text-muted mb-4">
            <p>MoodTunes &copy; 2025 - Advanced Music Recommendations</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>""")
    
    # Index template
    with open('templates/index.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Personalized Recommendations{% endblock %}

{% block content %}
<div class="header">
    <h1>MoodTunes</h1>
    <p class="lead">AI-powered music recommendations based on your mood, preferences, and feedback</p>
</div>

{% if loading.is_loading %}
<div class="card">
    <div class="card-body text-center">
        <h3>System Initializing</h3>
        <p>{{ loading.status }}</p>
        <div class="progress mb-3">
            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                 role="progressbar" 
                 style="width: {{ loading.progress }}%" 
                 aria-valuenow="{{ loading.progress }}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
                {{ loading.progress }}%
            </div>
        </div>
        <p>Please wait while we load the recommendation system...</p>
    </div>
</div>
{% else %}
<div class="row">
    <div class="col-md-6">
        <div class="card h-100">
            <div class="card-body">
                <h2 class="card-title">How It Works</h2>
                <p class="card-text">
                    MoodTunes uses advanced AI to understand your musical preferences. 
                    Just tell us how you're feeling or what you're doing, and we'll create 
                    the perfect playlist for you.
                </p>
                <h5>Features:</h5>
                <ul>
                    <li>Natural language understanding of your preferences</li>
                    <li>Cross-genre recommendations based on mood</li>
                    <li>Learns from your feedback to get better over time</li>
                    <li>Creates personalized playlists that match your unique taste</li>
                </ul>
                
                {% if not user_id %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('login') }}" class="btn btn-primary btn-lg">
                        <i class="fas fa-sign-in-alt me-2"></i>Get Started
                    </a>
                </div>
                {% else %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('new_recommendations') }}" class="btn btn-primary btn-lg">
                        <i class="fas fa-music me-2"></i>Get Recommendations
                    </a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card h-100">
            <div class="card-body">
                <h2 class="card-title">How To Use</h2>
                <div class="mb-4">
                    <h5><i class="fas fa-edit me-2"></i>1. Tell us how you feel</h5>
                    <p>Describe your mood, activity, or preferences in natural language.</p>
                </div>
                
                <div class="mb-4">
                    <h5><i class="fas fa-headphones me-2"></i>2. Explore recommendations</h5>
                    <p>Listen to personalized song recommendations tailored to your input.</p>
                </div>
                
                <div class="mb-4">
                    <h5><i class="fas fa-star me-2"></i>3. Provide feedback</h5>
                    <p>Rate songs to help the system learn your preferences.</p>
                </div>
                
                <div>
                    <h5><i class="fas fa-magic me-2"></i>4. Get your perfect playlist</h5>
                    <p>The system creates a customized playlist that gets better the more you use it.</p>
                </div>
                
                {% if user_id %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('dashboard') }}" class="btn btn-outline-primary btn-lg">
                        <i class="fas fa-columns me-2"></i>Go to Dashboard
                    </a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endif %}
{% endblock %}

{% block scripts %}
<script>
    // Check loading status periodically if system is initializing
    {% if loading.is_loading %}
    function checkStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (!data.is_loading) {
                    // Reload the page when loading is complete
                    window.location.reload();
                } else {
                    // Update the loading status
                    document.querySelector('.progress-bar').style.width = data.progress + '%';
                    document.querySelector('.progress-bar').setAttribute('aria-valuenow', data.progress);
                    document.querySelector('.progress-bar').textContent = data.progress + '%';
                    document.querySelector('p').textContent = data.status;
                    
                    // Check again in 2 seconds
                    setTimeout(checkStatus, 2000);
                }
            });
    }
    
    // Start checking status
    setTimeout(checkStatus, 2000);
    {% endif %}
</script>
{% endblock %}""")
    
    # Login template
    with open('templates/login.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Login{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-dark text-white">
                <h3 class="mb-0">Welcome to MoodTunes</h3>
            </div>
            <div class="card-body">
                {% if error %}
                <div class="alert alert-danger">{{ error }}</div>
                {% endif %}
                
                <ul class="nav nav-tabs" id="loginTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="new-user-tab" data-bs-toggle="tab" 
                                data-bs-target="#new-user" type="button" role="tab" 
                                aria-controls="new-user" aria-selected="true">
                            New User
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="existing-user-tab" data-bs-toggle="tab" 
                                data-bs-target="#existing-user" type="button" role="tab" 
                                aria-controls="existing-user" aria-selected="false">
                            Existing User
                        </button>
                    </li>
                </ul>
                
                <div class="tab-content mt-3" id="loginTabsContent">
                    <div class="tab-pane fade show active" id="new-user" role="tabpanel" 
                         aria-labelledby="new-user-tab">
                        <form method="post" action="{{ url_for('login') }}">
                            <input type="hidden" name="action" value="new_user">
                            
                            <div class="mb-3">
                                <label for="user_name" class="form-label">Your Name (Optional)</label>
                                <input type="text" class="form-control" id="user_name" name="user_name" 
                                       placeholder="Enter your name">
                                <div class="form-text">
                                    A unique user ID will be generated for you automatically.
                                </div>
                            </div>
                            
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-user-plus me-2"></i>Create New User
                                </button>
                            </div>
                        </form>
                    </div>
                    
                    <div class="tab-pane fade" id="existing-user" role="tabpanel" 
                         aria-labelledby="existing-user-tab">
                        <form method="post" action="{{ url_for('login') }}">
                            <input type="hidden" name="action" value="login">
                            
                            <div class="mb-3">
                                <label for="user_id" class="form-label">User ID</label>
                                <input type="text" class="form-control" id="user_id" name="user_id" 
                                       placeholder="Enter your user ID" required>
                                <div class="form-text">
                                    Enter the user ID that was provided to you previously.
                                </div>
                            </div>
                            
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-sign-in-alt me-2"></i>Login
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}""")
    
    # Dashboard template
    with open('templates/dashboard.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Dashboard{% endblock %}

{% block content %}
<div class="header">
    <h1>Your Music Dashboard</h1>
    <p class="lead">Welcome back, {{ user_name }}</p>
</div>

<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="card-title">
                    <i class="fas fa-headphones-alt fa-2x d-block mb-2 text-primary"></i>
                    {{ stats.recommendation_count }}
                </h3>
                <p class="card-text text-muted">Songs Recommended</p>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="card-title">
                    <i class="fas fa-star fa-2x d-block mb-2 text-warning"></i>
                    {{ stats.feedback_count }}
                </h3>
                <p class="card-text text-muted">Ratings Provided</p>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="card-title">
                    <i class="fas fa-music fa-2x d-block mb-2 text-success"></i>
                    {{ stats.high_rated_count }}
                </h3>
                <p class="card-text text-muted">Favorite Songs</p>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0">Your Music Taste</h4>
            </div>
            <div class="card-body">
                {% if stats.top_genres %}
                <h5>Top Genres You Like</h5>
                <div class="mb-4">
                    {% for genre, score in stats.top_genres %}
                    <div class="mb-2">
                        <div class="d-flex justify-content-between mb-1">
                            <span>{{ genre }}</span>
                            <span>{{ (score * 20)|int }}%</span>
                        </div>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: {{ (score * 20)|int }}%"></div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <p class="text-muted">Listen to more music to see your genre preferences.</p>
                {% endif %}
                
                {% if stats.top_moods %}
                <h5 class="mt-4">Your Mood Preferences</h5>
                <div class="d-flex flex-wrap">
                    {% for mood, score in stats.top_moods %}
                    <div class="bg-light rounded-pill px-3 py-1 me-2 mb-2">
                        {{ mood }} <span class="badge bg-primary ms-1">{{ (score * 20)|int }}%</span>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if stats.top_artists %}
                <h5 class="mt-4">Artists You Like</h5>
                <ul class="list-group">
                    {% for artist, score in stats.top_artists %}
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        {{ artist }}
                        <span class="badge bg-primary rounded-pill">{{ (score * 20)|int }}%</span>
                    </li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card mb-4">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0">Quick Actions</h4>
            </div>
            <div class="card-body">
                <div class="d-grid gap-3">
                    <a href="{{ url_for('new_recommendations') }}" class="btn btn-lg btn-primary">
                        <i class="fas fa-magic me-2"></i>Get New Recommendations
                    </a>
                    <a href="{{ url_for('final_playlist') }}" class="btn btn-lg btn-outline-primary">
                        <i class="fas fa-list me-2"></i>View Your Personalized Playlist
                    </a>
                    <a href="#" class="btn btn-lg btn-outline-secondary" data-bs-toggle="modal" data-bs-target="#userIdModal">
                        <i class="fas fa-id-card me-2"></i>View Your User ID
                    </a>
                </div>
            </div>
        </div>
        
        {% if stats.session_count > 0 %}
        <div class="card">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0">System Insights</h4>
            </div>
            <div class="card-body">
                <p>
                    <i class="fas fa-brain text-primary me-2"></i>
                    The recommendation system is learning your preferences based on 
                    {{ stats.feedback_count }} ratings across {{ stats.session_count }} sessions.
                </p>
                
                {% if stats.average_rating > 0 %}
                <p>
                    <i class="fas fa-chart-line text-success me-2"></i>
                    Your average rating is {{ "%.1f"|format(stats.average_rating) }}/5, which helps 
                    the system find music you'll love.
                </p>
                {% endif %}
                
                <p>
                    <i class="fas fa-lightbulb text-warning me-2"></i>
                    Tip: Rate more songs to improve your recommendations!
                </p>
            </div>
        </div>
        {% endif %}
    </div>
</div>

<!-- User ID Modal -->
<div class="modal fade" id="userIdModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Your User ID</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <p>Keep your User ID to login again in the future:</p>
                <div class="alert alert-info">
                    <span id="userId">{{ user_id }}</span>
                    <button class="btn btn-sm btn-outline-primary float-end" onclick="copyUserId()">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    function copyUserId() {
        const userId = document.getElementById('userId').textContent;
        navigator.clipboard.writeText(userId).then(() => {
            alert('User ID copied to clipboard!');
        });
    }
</script>
{% endblock %}""")
    
    # New Recommendations template
    with open('templates/new_recommendations.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - New Recommendations{% endblock %}

{% block content %}
<div class="header">
    <h1>Get Music Recommendations</h1>
    <p class="lead">Tell us how you're feeling or what you're doing</p>
</div>

<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card">
            <div class="card-body">
                <form method="post" action="{{ url_for('new_recommendations') }}">
                    {% for section in form.sections %}
                    <div class="mb-4">
                        <h4>{{ section.title }}</h4>
                        {% if section.description %}
                        <p class="text-muted">{{ section.description }}</p>
                        {% endif %}
                        
                        {% for field in section.fields %}
                        <div class="mb-3">
                            <label for="{{ field.id }}" class="form-label">{{ field.label }}</label>
                            
                            {% if field.type == 'text_area' %}
                            <textarea class="form-control" id="{{ field.id }}" name="{{ field.id }}" 
                                     rows="4" placeholder="{{ field.placeholder }}"
                                     {% if field.required %}required{% endif %}
                                     {% if field.min_length %}minlength="{{ field.min_length }}"{% endif %}
                                     {% if field.max_length %}maxlength="{{ field.max_length }}"{% endif %}></textarea>
                            
                            {% elif field.type == 'text' %}
                            <input type="text" class="form-control" id="{{ field.id }}" name="{{ field.id }}" 
                                   placeholder="{{ field.placeholder }}"
                                   {% if field.required %}required{% endif %}>
                            
                            {% elif field.type == 'select' %}
                            <select class="form-select" id="{{ field.id }}" name="{{ field.id }}"
                                   {% if field.required %}required{% endif %}>
                                <option value="" selected disabled>Select an option</option>
                                {% for option in field.options %}
                                <option value="{{ option }}" {% if field.default == option %}selected{% endif %}>
                                    {{ option }}
                                </option>
                                {% endfor %}
                            </select>
                            
                            {% elif field.type == 'multi_select' %}
                            <select class="form-select" id="{{ field.id }}" name="{{ field.id }}" 
                                   multiple size="5"
                                   {% if field.required %}required{% endif %}>
                                {% for option in field.options %}
                                <option value="{{ option }}">{{ option }}</option>
                                {% endfor %}
                            </select>
                            {% if field.max_selections %}
                            <div class="form-text">Select up to {{ field.max_selections }} options</div>
                            {% endif %}
                            
                            {% elif field.type == 'slider' %}
                            <div class="range-container">
                                <input type="range" class="form-range" min="{{ field.min }}" max="{{ field.max }}"
                                      id="{{ field.id }}" name="{{ field.id }}" value="{{ field.default }}">
                                <div class="d-flex justify-content-between">
                                    <span>Similar</span>
                                    <span>Diverse</span>
                                </div>
                            </div>
                            {% if field.help_text %}
                            <div class="form-text">{{ field.help_text }}</div>
                            {% endif %}
                            
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                    
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-magic me-2"></i>Generate Recommendations
                        </button>
                        <button type="reset" class="btn btn-outline-secondary">
                            <i class="fas fa-undo me-2"></i>Reset Form
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}""")
    
    # Recommendations template
    with open('templates/recommendations.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Your Recommendations{% endblock %}

{% block content %}
<div class="header">
    <h1>Your Recommendations</h1>
    <p class="lead">Rate songs to improve future recommendations</p>
</div>

<div class="row mb-4">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center">
            <h2>We found {{ recommendations|length }} songs for you</h2>
            <div>
                <a href="{{ url_for('new_recommendations') }}" class="btn btn-outline-primary">
                    <i class="fas fa-sync-alt me-2"></i>New Recommendations
                </a>
                <a href="{{ url_for('final_playlist') }}" class="btn btn-primary ms-2">
                    <i class="fas fa-list me-2"></i>Generate My Playlist
                </a>
            </div>
        </div>
    </div>
</div>

<div class="row">
    {% for video in recommendations %}
    <div class="col-md-6 col-lg-4 mb-4">
        <div class="card h-100">
            <div class="video-container">
                <div class="recommendation-method">
                    {% if video.recommendation_method == 'text' %}
                    <i class="fas fa-comment me-1"></i> From Your Text
                    {% elif video.recommendation_method == 'mood' %}
                    <i class="fas fa-smile me-1"></i> Mood Match
                    {% elif video.recommendation_method == 'genre_mix' %}
                    <i class="fas fa-music me-1"></i> Genre Mix
                    {% elif video.recommendation_method == 'popular' %}
                    <i class="fas fa-fire me-1"></i> Popular
                    {% else %}
                    <i class="fas fa-random me-1"></i> Discover
                    {% endif %}
                </div>
                <iframe src="{{ video.embed_url }}" title="{{ video.title }}"
                        frameborder="0" allowfullscreen></iframe>
            </div>
            <div class="card-body">
                <h5 class="card-title text-truncate" title="{{ video.title }}">
                    {{ video.title }}
                </h5>
                <p class="card-text text-muted">{{ video.artist }}</p>
                <div class="d-grid">
                    <a href="{{ url_for('feedback', video_id=video.video_id) }}" class="btn btn-primary">
                        <i class="fas fa-star me-2"></i>Rate This Song
                    </a>
                </div>
            </div>
            <div class="card-footer text-muted d-flex justify-content-between align-items-center">
                <small>
                    <i class="fas fa-tag me-1"></i> {{ video.genre }}
                </small>
                <small>
                    <i class="fas fa-eye me-1"></i> {{ '{:,}'.format(video.views) }}
                </small>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% endblock %}""")
    
    # Feedback template
    with open('templates/feedback.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Rate This Song{% endblock %}

{% block content %}
<div class="header">
    <h1>Rate This Song</h1>
    <p class="lead">Your feedback helps us improve your recommendations</p>
</div>

<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card mb-4">
            <div class="card-body">
                <div class="video-container mb-3">
                    <iframe src="{{ video.embed_url }}" title="{{ video.title }}"
                            frameborder="0" allowfullscreen></iframe>
                </div>
                <h4 class="card-title">{{ video.title }}</h4>
                <p class="card-text text-muted">{{ video.artist }}</p>
                
                <div class="d-flex justify-content-between align-items-center">
                    <span class="badge bg-primary">{{ video.genre }}</span>
                    <span class="text-muted">
                        <i class="fas fa-eye me-1"></i>{{ '{:,}'.format(video.views) }} views
                    </span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0">Your Rating</h4>
            </div>
            <div class="card-body">
                <form method="post" action="{{ url_for('feedback', video_id=video.video_id) }}"
                      id="feedbackForm">
                      
                    <div class="form-group mb-4">
                        <label for="rating" class="form-label">{{ form.fields[0].label }}</label>
                        <div class="rating">
                            <input type="radio" id="star5" name="rating" value="5" required />
                            <label for="star5" title="5 stars"><i class="fas fa-star"></i></label>
                            
                            <input type="radio" id="star4" name="rating" value="4" />
                            <label for="star4" title="4 stars"><i class="fas fa-star"></i></label>
                            
                            <input type="radio" id="star3" name="rating" value="3" />
                            <label for="star3" title="3 stars"><i class="fas fa-star"></i></label>
                            
                            <input type="radio" id="star2" name="rating" value="2" />
                            <label for="star2" title="2 stars"><i class="fas fa-star"></i></label>
                            
                            <input type="radio" id="star1" name="rating" value="1" />
                            <label for="star1" title="1 star"><i class="fas fa-star"></i></label>
                        </div>
                    </div>
                    
                    <input type="hidden" id="listen_duration" name="listen_duration" value="0" />
                    
                    <div class="form-group mb-4">
                        <label for="skip_reason" class="form-label">{{ form.fields[2].label }}</label>
                        <select class="form-select" id="skip_reason" name="skip_reason">
                            {% for option in form.fields[2].options %}
                            <option value="{{ option }}" {% if form.fields[2].default == option %}selected{% endif %}>
                                {{ option }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group mb-4">
                        <label for="comments" class="form-label">{{ form.fields[3].label }}</label>
                        <textarea class="form-control" id="comments" name="comments" rows="3"
                                 placeholder="{{ form.fields[3].placeholder }}"></textarea>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-paper-plane me-2"></i>Submit Rating
                        </button>
                        <a href="{{ url_for('view_recommendations') }}" class="btn btn-outline-secondary">
                            <i class="fas fa-arrow-left me-2"></i>Back to Recommendations
                        </a>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // Track how long the user listens to the song
    let startTime = Date.now();
    let isPlaying = true;
    
    // Update the listen duration on form submit
    document.getElementById('feedbackForm').addEventListener('submit', function() {
        const duration = isPlaying ? (Date.now() - startTime) / 1000 : 0;
        document.getElementById('listen_duration').value = duration;
    });
    
    // Show skip reason only for low ratings
    document.querySelectorAll('input[name="rating"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const rating = parseInt(this.value);
            const skipReasonField = document.getElementById('skip_reason').parentNode;
            
            if (rating <= 2) {
                skipReasonField.style.display = 'block';
            } else {
                skipReasonField.style.display = 'none';
                document.getElementById('skip_reason').value = 'Not applicable';
            }
        });
    });
    
    // Hide skip reason by default
    document.getElementById('skip_reason').parentNode.style.display = 'none';
</script>
{% endblock %}""")
    
    # Final Playlist template
    with open('templates/final_playlist.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Your Personalized Playlist{% endblock %}

{% block content %}
<div class="header">
    <h1>{{ playlist.name }}</h1>
    <p class="lead">{{ playlist.description }}</p>
</div>

<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h4 class="mb-1">{{ playlist.size }} songs</h4>
                        <p class="text-muted mb-0">
                            <i class="fas fa-calendar-alt me-1"></i>Created {{ playlist.created_at|format_date }}
                        </p>
                    </div>
                    <div>
                        <a href="{{ url_for('new_recommendations') }}" class="btn btn-outline-primary">
                            <i class="fas fa-magic me-2"></i>Get New Recommendations
                        </a>
                    </div>
                </div>
                
                {% if playlist.genre_distribution %}
                <div class="mt-4">
                    <h5>Genre Distribution</h5>
                    <div class="row">
                        {% for genre, percentage in playlist.genre_distribution.items() %}
                        <div class="col-md-4 col-sm-6 mb-2">
                            <div class="d-flex justify-content-between mb-1">
                                <span>{{ genre }}</span>
                                <span>{{ (percentage * 100)|int }}%</span>
                            </div>
                            <div class="progress" style="height: 8px;">
                                <div class="progress-bar" role="progressbar" 
                                     style="width: {{ (percentage * 100)|int }}%"></div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<div class="row">
    {% for track in playlist.tracks %}
    <div class="col-md-6 col-lg-4 mb-4">
        <div class="card h-100">
            <div class="video-container">
                <iframe src="{{ track.embed_url }}" title="{{ track.title }}"
                        frameborder="0" allowfullscreen></iframe>
                {% if 'user_rating' in track %}
                <div class="recommendation-method">
                    <i class="fas fa-star me-1 text-warning"></i> Rated {{ track.user_rating }}/5
                </div>
                {% endif %}
            </div>
            <div class="card-body">
                <h5 class="card-title text-truncate" title="{{ track.title }}">
                    {{ track.title }}
                </h5>
                <p class="card-text text-muted">{{ track.artist }}</p>
                <a href="{{ track.youtube_url }}" target="_blank" class="btn btn-sm btn-outline-primary w-100">
                    <i class="fab fa-youtube me-1"></i> Watch on YouTube
                </a>
            </div>
            <div class="card-footer text-muted d-flex justify-content-between align-items-center">
                <small>
                    <i class="fas fa-tag me-1"></i> {{ track.genre }}
                </small>
                <small>
                    <i class="fas fa-eye me-1"></i> {{ '{:,}'.format(track.views) }}
                </small>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% endblock %}""")
    
    # Loading template
    with open('templates/loading.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Loading{% endblock %}

{% block content %}
<div class="card">
    <div class="card-body text-center py-5">
        <h3>System Initializing</h3>
        <p>{{ loading.status }}</p>
        <div class="progress mb-3">
            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                 role="progressbar" 
                 style="width: {{ loading.progress }}%" 
                 aria-valuenow="{{ loading.progress }}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
                {{ loading.progress }}%
            </div>
        </div>
        <p>Please wait while we load the recommendation system...</p>
        <div class="spinner-border text-primary mt-3" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // Check loading status periodically
    function checkStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (!data.is_loading) {
                    // Redirect when loading is complete
                    window.location.href = "{{ redirect_url }}";
                } else {
                    // Update the loading status
                    document.querySelector('.progress-bar').style.width = data.progress + '%';
                    document.querySelector('.progress-bar').setAttribute('aria-valuenow', data.progress);
                    document.querySelector('.progress-bar').textContent = data.progress + '%';
                    document.querySelector('p').textContent = data.status;
                    
                    // Check again in 2 seconds
                    setTimeout(checkStatus, 2000);
                }
            });
    }
    
    // Start checking status
    setTimeout(checkStatus, 2000);
</script>
{% endblock %}""")
    
    # Error template
    with open('templates/error.html', 'w') as f:
        f.write("""{% extends 'base.html' %}

{% block title %}MoodTunes - Error{% endblock %}

{% block content %}
<div class="card text-center">
    <div class="card-header bg-danger text-white">
        <h3 class="mb-0">Oops! Something went wrong</h3>
    </div>
    <div class="card-body py-5">
        <i class="fas fa-exclamation-triangle text-danger fa-4x mb-4"></i>
        <h4>{{ error }}</h4>
        <p class="text-muted">We apologize for the inconvenience.</p>
        <div class="mt-4">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-primary me-2">
                <i class="fas fa-home me-2"></i>Back to Dashboard
            </a>
            <a href="{{ url_for('new_recommendations') }}" class="btn btn-primary">
                <i class="fas fa-sync-alt me-2"></i>Try Again
            </a>
        </div>
    </div>
</div>
{% endblock %}""")

# Run the application
if __name__ == '__main__':
    # Create the templates
    create_templates()
    
    # Run the application
    app.run(debug=True)