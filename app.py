# ✅ MUST BE FIRST - Disable SSL verification for development
import os
import ssl
import urllib3
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from etl_pipeline.real_api_coordinator import RealAPICoordinator
from ml_models.sprint_health_predictor import SprintHealthPredictor
from ml_models.productivity_heatmap_ai import ProductivityHeatmapAI
from ml_models.dependency_tracker import DependencyTracker
from processing.processing_coordinator import ProcessingCoordinator
from etl_pipeline.etl_coordinator import ETLCoordinator
from flask import Flask, jsonify,request, g
from flask_sqlalchemy import SQLAlchemy
from config import Config
from models import db, SprintHealthMetrics, ProductivityPatterns, DependencyChains, User, Task, Project, ProjectMember, Notification, CalendarData, CommunicationData, CodeActivity
from flask import send_file, render_template_string
from flask_cors import CORS
from models import User
# ✅ Disable OAuth redirects during development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Allow HTTP for localhost

# ✅ Global flag to check if we're in the middle of OAuth
OAUTH_IN_PROGRESS = False
from datetime import datetime
from notification_service import NotificationService, AutoAlertService
from processing.time_series_aggregation import TimeSeriesAggregator
from processing.calendar_density_analysis import CalendarDensityAnalyzer
from processing.redis_cache import RedisCache
import logging

# Add this right after your imports
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




# Email Configuration
EMAIL_SENDER = "roobabaskaran194@gmail.com"  # Your Gmail
EMAIL_PASSWORD = "pyflxnrrbpfssvwh"    # Gmail App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
#from redis_cache import RedisCache

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    # ✅ ADD THIS LINE AFTER app creation
    CORS(app)

    # ✅ Add this to prevent OAuth redirects
    @app.before_request
    def prevent_oauth_loops():
        """Prevent OAuth redirect loops during normal navigation"""
        # If navigating to /oauth2callback without proper state
        if request.path == '/oauth2callback':
            if not request.args.get('code') and not request.args.get('state'):
                logger.info("⚠️ Caught stray OAuth redirect, sending to dashboard")
                return redirect('/manager-dashboard')
        return None


    

    # Database configuration
    DATABASE = 'agile_nexus.db'

    # ✅ NOW DEFINE get_db() AFTER app is created
    def get_db():
        """Get database connection"""
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
            db.row_factory = sqlite3.Row
        return db

    # ✅ NOW USE @app decorator
    @app.teardown_appcontext
    def close_connection(exception):
        """Close database connection at end of request"""
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    
    # Initialize database
    db.init_app(app)
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    # ==================== AUTHENTICATION ROUTES ====================
    
    @app.route('/register', methods=['POST'])  # ✅ REMOVE /api prefix
    def register():
        """Register new user"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['username', 'email', 'password', 'full_name', 'role']
            for field in required_fields:
                if field not in data or not data[field]:
                    return jsonify({'status': 'error', 'error': f'{field} is required'}), 400
            
            # Check if username exists
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'status': 'error', 'error': 'Username already exists'}), 400
            
            # Check if email exists
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'status': 'error', 'error': 'Email already exists'}), 400
            
            # Validate role
            if data['role'] not in ['manager', 'member']:
                return jsonify({'status': 'error', 'error': 'Invalid role. Must be manager or member'}), 400
            
            # Create new user
            new_user = User(
                username=data['username'],
                email=data['email'],
                full_name=data['full_name'],
                role=data['role'],
                created_at=datetime.utcnow()
            )
            new_user.set_password(data['password'])
            
            db.session.add(new_user)
            db.session.commit()
            
            return jsonify({
                'status': 'success',  # ✅ ADD status field
                'message': 'Registration successful',
                'user': new_user.to_dict()
            }), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/login', methods=['POST'])  # ✅ REMOVE /api prefix
    def login():
        """User login"""
        try:
            data = request.get_json()
            
            # Validate
            if not data.get('username') or not data.get('password'):
                return jsonify({'status': 'error', 'error': 'Username and password required'}), 400
            
            # Find user
            user = User.query.filter_by(username=data['username']).first()
            
            if not user or not user.check_password(data['password']):
                return jsonify({'status': 'error', 'error': 'Invalid username or password'}), 401
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'status': 'success',  # ✅ ADD status field
                'message': 'Login successful',
                'user': user.to_dict()
            }), 200
            
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    
    @app.route('/api/user/<username>', methods=['GET'])
    def get_user(username):
        """Get user details"""
        try:
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            return jsonify({'user': user.to_dict()}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # ==================== FRONTEND ROUTES ====================
    
    @app.route('/')
    def index():
        """Serve login page"""
        return send_file('../frontend/templates/login.html')  # ✅ Adjust path to your structure
    
    @app.route('/manager-dashboard')
    def manager_dashboard():
        """Serve manager dashboard"""
        return send_file('../frontend/templates/manager_dashboard.html')
    
    @app.route('/team-dashboard')
    def team_dashboard():
        """Serve team member dashboard"""
        return send_file('../frontend/templates/member_dashboard.html')
    
    @app.route('/risk-details')
    def risk_details_page():
        """Serve risk details page"""
        return send_file('../frontend/templates/risk_details.html')

    @app.route('/productivity-details')
    def productivity_details_page():
        """Serve productivity details page"""
        return send_file('../frontend/templates/productivity_details.html')
    
    @app.route('/dependency-details')
    def dependency_details_page():
        """Serve dependency details page"""
        return send_file('../frontend/templates/dependency_details.html')
    
    @app.route('/team-management')
    def team_management_page():
        """Serve team management page"""
        return send_file('../frontend/templates/team_management.html')
    @app.route('/data-verification')
    def data_verification_page():
        """Serve data verification page"""
        return send_file('../frontend/templates/data_verification.html')

    @app.route('/oauth2callback')
    def oauth2callback():
        """Handle OAuth callback and redirect to dashboard"""
        # Store any OAuth tokens if needed
        """Handle OAuth callback silently"""
        # Check if this is a real OAuth callback with a code
        if request.args.get('code'):
            # Process the OAuth code here if needed
            pass
    
        # ✅ Always redirect back to dashboard without showing message
        return redirect('/manager-dashboard')
    @app.before_request
    def check_oauth_redirect():
        """Prevent OAuth callback redirects during normal navigation"""
        if request.path == '/oauth2callback' and not request.args.get('code'):
            # This is a stray redirect, send back to dashboard
            return redirect('/manager-dashboard')

    @app.route('/high-risk-users')
    def high_risk_users_page():
        """Serve high risk users page"""
        return send_file('../frontend/templates/high_risk_users.html')
    
    @app.route('/task-assignment')
    def task_assignment_page():
        """Serve task assignment page"""
        return send_file('../frontend/templates/task_assignment.html')

    @app.route('/task-tracker')
    def task_tracker_page():
        """Serve task tracker page"""
        return send_file('../frontend/templates/task_tracker.html')

    @app.route('/member-risk-details')
    def member_risk_details_page():
        """Serve member risk details page"""
        return send_file('../frontend/templates/member_risk_details.html')

    @app.route('/member-productivity-details')
    def member_productivity_details_page():
        """Serve member productivity details page"""
        return send_file('../frontend/templates/member_productivity_details.html')
    @app.route('/member-risk-analysis')
    def member_risk_analysis():
        return send_file('../frontend/templates/member_risk_analysis.html')
    # ==================== BASIC ROUTES (Parts 1-2) ====================
    

    @app.route('/create-project')
    def create_project_page():
        """Serve create project page"""
        return send_file('../frontend/templates/create_project.html')

    @app.route('/test-db')
    def test_db():
        try:
            result = db.session.execute(db.text('SELECT version()'))
            version = result.fetchone()[0]
            db.session.close()
            
            return jsonify({
                "database": "connected", 
                "status": "success",
                "version": version
            })
        except Exception as e:
            return jsonify({
                "database": "failed", 
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/test-insert')
    def test_insert():
        try:
            test_metric = SprintHealthMetrics(
                sprint_id="TEST_SPRINT_001",
                team_id="DEV_TEAM_01",
                calendar_density=0.7,
                meeting_hours=4.5,
                focus_time_hours=3.5,
                predicted_risk_score=35.0,
                risk_factors='{"high_meetings": true, "low_focus_time": false}'
            )
            
            db.session.add(test_metric)
            db.session.commit()
            
            return jsonify({
                "message": "Test data inserted successfully!",
                "status": "success"
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    # ==================== ETL PIPELINE (Part 3) ====================
    
    @app.route('/run-etl')
    def run_etl():
        try:
            etl = ETLCoordinator(app.config)
            results = etl.run_full_extraction(days_back=7)
            
            return jsonify({
                "message": "ETL extraction completed successfully!",
                "results": results,
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "ETL extraction failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    # ==================== PROCESSING PIPELINE (Part 4) ====================
    
    @app.route('/run-processing')
    def run_processing():
        try:
            processor = ProcessingCoordinator()
            results = processor.run_full_processing(days_back=7, include_mock=False)
            
            return jsonify({
                "message": "Data processing completed successfully!",
                "results": results,
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "Data processing failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    # ✅ ADD NEW ROUTE to toggle mock data
    @app.route('/run-processing-all')
    def run_processing_all():
        """Include both real and mock data"""
        try:
            processor = ProcessingCoordinator()
            results = processor.run_full_processing(days_back=7, include_mock=True)  # ✅ Include mock
        
            return jsonify({
                "message": "Data processing completed (with mock data)!",
                "results": results,
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "Data processing failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    # ==================== SPRINT HEALTH PREDICTOR (Part 5A) ====================
    
    @app.route('/test-sprint-model')
    def test_sprint_model():
        try:
            training_data = [
                {
                    # ✅ Use your real team members instead of test_user
                    'user_id': 'rooba8925',
                    'meeting_hours': 3.0,
                    'focus_hours': 4.0,
                    'total_messages': 20,
                    'total_commits': 10,
                    'calendar_density': 0.6,
                    'productivity_score': 0.8,
                    'risk_score': 15
                },
                {
                    'user_id': 'praneetaad078',
                    'meeting_hours': 4.5,
                    'focus_hours': 3.0,
                    'total_messages': 30,
                    'total_commits': 6,
                    'calendar_density': 0.7,
                    'productivity_score': 0.6,
                    'risk_score': 45
                },
                {
                    'user_id': 'sadhana-095',
                    'meeting_hours': 3.5,
                    'focus_hours': 3.5,
                    'total_messages': 25,
                    'total_commits': 2,
                    'calendar_density': 0.65,
                    'productivity_score': 0.7,
                    'risk_score': 30
                }
            ]
            
            predictor = SprintHealthPredictor()
            training_result = predictor.train(training_data)
            
            test_data = [
                {
                    'user_id': 'rooba8925',
                    'meeting_hours': 3.5,
                    'focus_hours': 4.5,
                    'total_messages': 22,
                    'total_commits': 10,
                    'calendar_density': 0.65,
                    'productivity_score': 0.82
                },
                {
                    'user_id': 'praneetaad078',
                    'meeting_hours': 5.0,
                    'focus_hours': 2.5,
                    'total_messages': 35,
                    'total_commits': 6,
                    'calendar_density': 0.75,
                    'productivity_score': 0.58
                },
                {
                    'user_id': 'sadhana-095',
                    'meeting_hours': 4.0,
                    'focus_hours': 3.0,
                    'total_messages': 28,
                    'total_commits': 2,
                    'calendar_density': 0.70,
                    'productivity_score': 0.65
                }
            ]
            
            predictions = predictor.predict(test_data)
            
            return jsonify({
                "message": "Sprint Health Model test completed!",
                "training_result": training_result,
                "test_predictions": predictions,
                "status": "success"
            })
            
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Sprint model test failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    @app.route('/api/productivity/<username>', methods=['GET'])
    def get_member_productivity(username):
        """
        Single source of truth for member productivity calculation.
        Returns consistent productivity score across all dashboards.
        """
        try:
            # Get all tasks for this user
            tasks = Task.query.filter_by(assigned_to=username).all()
        
            if not tasks or len(tasks) == 0:
                # No tasks yet - return baseline productivity
                return jsonify({
                    'status': 'success',
                    'productivity': 70,
                    'completion_rate': 0,
                    'total_tasks': 0,
                    'completed_tasks': 0,
                    'in_progress_tasks': 0,
                    'pending_tasks': 0,
                    'message': 'No tasks assigned yet'
                })
        
            # Calculate task statistics
            total = len(tasks)
            completed = len([t for t in tasks if t.status == 'completed'])
            in_progress = len([t for t in tasks if t.status == 'in_progress'])
            pending = len([t for t in tasks if t.status == 'pending'])
        
            # Calculate completion rate
            completion_rate = round((completed / total * 100)) if total > 0 else 0
        
            # Calculate activity factor (how many tasks are being actively worked on)
            activity_factor = min(100, (in_progress + completed) * 20)
        
            # Calculate final productivity score
            # Formula: 70% weight on completion rate + 30% weight on activity
            productivity = round((completion_rate * 0.7) + (activity_factor * 0.3))
        
            return jsonify({
                'status': 'success',
                'productivity': productivity,
                'completion_rate': completion_rate,
                'total_tasks': total,
                'completed_tasks': completed,
                'in_progress_tasks': in_progress,
                'pending_tasks': pending,
                'activity_factor': activity_factor
            })
        
        except Exception as e:
            print(f"Error calculating productivity for {username}: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'productivity': 0
            }), 500
            
    @app.route('/test-sprint-model-real')
    def test_sprint_model_real():
        try:
            from models import CodeActivity, CommunicationData, CalendarData, Task
        
            # ✅ Get real data from your database
            # ✅ Use actual GitHub usernames from database
            github_users_query = db.session.query(CodeActivity.user_id).filter(
                CodeActivity.data_source == 'real'
            ).distinct().all()

            github_users = [u[0] for u in github_users_query] if github_users_query else ['rooba8925', 'praneetaad078', 'sadhana-095']
        
            print(f"\n{'='*60}")
            print(f"🎯 BUILDING SPRINT MODEL WITH REAL DATA")
            print(f"{'='*60}")
            print(f"Users found: {github_users}")
        
            training_data = []
            for username in github_users:
                print(f"\n👤 Processing: {username}")

                # ✅ Create consistent seed for this user
                seed = sum(ord(c) for c in username.lower())
            
                def seeded_random(offset):
                    import math
                    x = math.sin(seed * 12345 + offset) * 10000
                    return x - math.floor(x)
                
                # Get GitHub data
                code_data = CodeActivity.query.filter_by(
                    user_id=username,
                    data_source='real'
                ).all()
            
                total_commits = sum(c.commits_count for c in code_data) if code_data else 0
                 # ✅ If no real commits, use consistent mock data
                if total_commits == 0:
                    total_commits = int(5 + seeded_random(3) * 20)  # 5-25 commits
            
                print(f"  📊 GitHub commits: {total_commits}")
            
                # Get Slack data
                comm_data = CommunicationData.query.filter_by(
                    data_source='real'
                ).order_by(CommunicationData.date.desc()).first()
            
                messages = comm_data.message_count if comm_data else 0
                # ✅ If no real messages, use consistent mock data
                if messages == 0:
                    messages = int(15 + seeded_random(4) * 35)  # 15-50 messages
                print(f"  💬 Slack messages: {messages}")
            
                # Get Calendar data
                cal_events = CalendarData.query.filter_by(
                    data_source='real'
                ).all()
            
                meeting_hours = min(len(cal_events) * 0.5, 8) if cal_events else 0
                # ✅ If no real calendar data, use consistent mock data
                if meeting_hours == 0:
                    meeting_hours = round(2 + seeded_random(1) * 3, 1)  # 2-5 hours
                
                focus_hours = max(8 - meeting_hours, 1)
                # ✅ Ensure focus hours has consistent mock data
                if focus_hours >= 7:  # If too high because meeting_hours was 0
                    focus_hours = round(3 + seeded_random(2) * 3, 1)  # 3-6 hours
                
                print(f"  📅 Calendar events: {len(cal_events)}")
                print(f"  ⏰ Meeting hours: {meeting_hours:.1f}h")
                print(f"  🎯 Focus hours: {focus_hours:.1f}h")
            
                # ✅ Calculate REAL productivity from tasks
                user_tasks = Task.query.filter_by(assigned_to=username).all()
            
                if user_tasks:
                    total_tasks = len(user_tasks)
                    completed = len([t for t in user_tasks if t.status == 'completed'])
                    in_progress = len([t for t in user_tasks if t.status == 'in_progress'])
                
                    completion_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
                    activity_factor = min(100, (in_progress + completed) * 20)
                    productivity_score = (completion_rate * 0.7 + activity_factor * 0.3) / 100

                
                    print(f"  ✅ Tasks: {completed}/{total_tasks} completed")
                    print(f"  📈 Productivity: {productivity_score*100:.1f}%")
                else:
                    # Fallback: calculate from commits and activity
                    commit_factor = min(1.0, total_commits / 10)  # Normalize to 0-1
                    meeting_factor = max(0, 1 - (meeting_hours / 8))  # Less meetings = more productive
                    productivity_score = (commit_factor * 0.6 + meeting_factor * 0.4)

                # ✅ Ensure productivity_score is between 0 and 1
                productivity_score = max(0.0, min(1.0, productivity_score))
                
                # Calculate risk score
                risk_score = 0
                if meeting_hours > 5:
                    risk_score += 30
                if focus_hours < 2:
                    risk_score += 25
                if total_commits < 5:
                    risk_score += 20
                if productivity_score < 0.5:
                    risk_score += 15
            
                print(f"  ⚠️ Risk score: {risk_score}%")
            
                training_data.append({
                    'user_id': username,
                    'meeting_hours': meeting_hours,
                    'focus_hours': focus_hours,
                    'total_messages': messages,
                    'total_commits': total_commits,
                    'calendar_density': min(meeting_hours / 8, 1.0),
                    'productivity_score': productivity_score,  # ✅ Now includes real productivity
                    'risk_score': risk_score
                })
            print(f"\n{'='*60}")
            print(f"✅ Training data prepared for {len(training_data)} users")
            print(f"{'='*60}\n")
        
            predictor = SprintHealthPredictor()
            training_result = predictor.train(training_data)
            predictions = predictor.predict(training_data)

            # ✅ ADD THIS AFTER training and predictions:
            # Enhance predictions with productivity_score
            if predictions and 'predictions' in predictions:
                for pred in predictions['predictions']:
                    # Add productivity score from original training data
                    matching_train = next(
                        (t for t in training_data if t['user_id'] == pred['user_id']), 
                        None
                    )
                    if matching_train:
                        pred['meeting_hours'] = matching_train.get('meeting_hours')
                        pred['focus_hours'] = matching_train.get('focus_hours')
                        pred['total_commits'] = matching_train.get('total_commits')
                        pred['total_messages'] = matching_train.get('total_messages')
                        pred['productivity_score'] = matching_train.get('productivity_score')
                    
            return jsonify({
                "message": "Sprint Health Model with REAL data!",
                "training_result": training_result,
                "test_predictions": predictions,
                "real_data_used": True,
                "data_sources": {
                    "github_users": len(github_users),
                    "total_commits": sum(d['total_commits'] for d in training_data),
                    "calendar_events": len(cal_events) if cal_events else 0,
                    "tasks_analyzed": sum(1 for d in training_data if d.get('productivity_score', 0) > 0)
                },
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Sprint model test failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500        
    
    @app.route('/train-sprint-model')
    def train_sprint_model():
        try:
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
            
            sprint_features = processing_results.get('sample_sprint_features', [])
            
            predictor = SprintHealthPredictor()
            training_result = predictor.train(sprint_features)
            predictor.save_model()
            
            return jsonify({
                "message": "Sprint Health Predictor trained successfully!",
                "training_result": training_result,
                "training_data_size": len(sprint_features),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "Sprint training failed",
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/predict-sprint-risk')
    def predict_sprint_risk():
        try:
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=3)
            
            sprint_features = processing_results.get('sample_sprint_features', [])
            
            if not sprint_features:
                sprint_features = [
                    {
                        'user_id': 'user_001',
                        'meeting_hours': 4.5,
                        'focus_hours': 2.0,
                        'total_messages': 25,
                        'total_commits': 3,
                        'calendar_density': 0.75,
                        'productivity_score': 0.65
                    },
                    {
                        'user_id': 'user_002',
                        'meeting_hours': 6.0,
                        'focus_hours': 1.5,
                        'total_messages': 35,
                        'total_commits': 1,
                        'calendar_density': 0.85,
                        'productivity_score': 0.45
                    },
                    {
                        'user_id': 'user_003',
                        'meeting_hours': 2.5,
                        'focus_hours': 5.0,
                        'total_messages': 15,
                        'total_commits': 7,
                        'calendar_density': 0.45,
                        'productivity_score': 0.85
                    }
                ]
            
            predictor = SprintHealthPredictor()
            # ✅ FIX: Train first, then predict
            training_result = predictor.train(sprint_features)
            predictions = predictor.predict(sprint_features)
            
            return jsonify({
                "message": "Sprint risk predictions generated!",
                "training_result": training_result,  # ✅ Add training info
                "predictions": predictions,
                "input_data_size": len(sprint_features),
                "status": "success"
            })
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Sprint prediction failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500
    
    # ==================== PRODUCTIVITY HEATMAP AI (Part 5B) ====================
    
    @app.route('/test-productivity-model')
    def test_productivity_model():
        try:
            training_data = [
                {
                    'user_id': 'user_001',
                    'hour_of_day': 10,
                    'day_of_week': 1,
                    'meeting_count': 0,
                    'focus_events': 2,
                    'calendar_density': 0.5,
                    'message_activity': 20,
                    'commit_activity': 4,
                    'productivity_score': 0.85
                },
                {
                    'user_id': 'user_002',
                    'hour_of_day': 15,
                    'day_of_week': 3,
                    'meeting_count': 3,
                    'focus_events': 0,
                    'calendar_density': 0.9,
                    'message_activity': 50,
                    'commit_activity': 1,
                    'productivity_score': 0.3
                },
                {
                    'user_id': 'user_003',
                    'hour_of_day': 9,
                    'day_of_week': 0,
                    'meeting_count': 1,
                    'focus_events': 2,
                    'calendar_density': 0.6,
                    'message_activity': 25,
                    'commit_activity': 3,
                    'productivity_score': 0.75
                }
            ]
            
            heatmap_ai = ProductivityHeatmapAI()
            training_result = heatmap_ai.train(training_data)
            
            test_data = [
                {
                    'user_id': 'test_user',
                    'hour_of_day': 11,
                    'day_of_week': 2,
                    'meeting_count': 1,
                    'focus_events': 1,
                    'calendar_density': 0.7,
                    'message_activity': 30,
                    'commit_activity': 2
                }
            ]
            
            predictions = heatmap_ai.predict(test_data)
            heatmap_data = heatmap_ai.get_team_heatmap()
            
            return jsonify({
                "message": "Productivity Heatmap AI test completed!",
                "training_result": training_result,
                "test_predictions": predictions,
                "team_heatmap": heatmap_data,
                "status": "success"
            })
            
        except Exception as e:
            return jsonify({
                "message": "Productivity model test failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/train-productivity-model')
    def train_productivity_model():
        try:
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
            
            productivity_features = processing_results.get('sample_productivity_features', [])
            
            heatmap_ai = ProductivityHeatmapAI()
            training_result = heatmap_ai.train(productivity_features)
            heatmap_ai.save_model()
            
            return jsonify({
                "message": "Productivity Heatmap AI trained successfully!",
                "training_result": training_result,
                "training_data_size": len(productivity_features),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "Productivity training failed",
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/predict-productivity')
    def predict_productivity():
        try:
            import os
        
            # ✅ Delete old model file
            model_path = 'data/productivity_heatmap_ai_model.pkl'
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"✅ Deleted old productivity model: {model_path}")
            sample_data = [
                {
                    'user_id': 'user_001',
                    'hour_of_day': 10,
                    'day_of_week': 1,
                    'meeting_count': 1,
                    'focus_events': 2,
                    'calendar_density': 0.6,
                    'message_activity': 25,
                    'commit_activity': 3,
                    'productivity_score': 0.85  # ✅ Add target for training
                },
                {
                    'user_id': 'user_002', 
                    'hour_of_day': 14,
                    'day_of_week': 2,
                    'meeting_count': 3,
                    'focus_events': 0,
                    'calendar_density': 0.9,
                    'message_activity': 45,
                    'commit_activity': 1,
                    'productivity_score': 0.45  # ✅ Add target
                },
                {
                    'user_id': 'user_003',
                    'hour_of_day': 9,
                    'day_of_week': 0,
                    'meeting_count': 0,
                    'focus_events': 3,
                    'calendar_density': 0.4,
                    'message_activity': 15,
                    'commit_activity': 5,
                    'productivity_score': 0.90  # ✅ Add target
                }
            ]
            
            heatmap_ai = ProductivityHeatmapAI()
            # ✅ FIX: Train first with targets, then predict
            training_result = heatmap_ai.train(sample_data)
            predictions = heatmap_ai.predict(sample_data)
            
            return jsonify({
                "message": "Productivity predictions generated!",
                "training_result": training_result,  # ✅ Add training info
                "predictions": predictions,
                "input_data_size": len(sample_data),
                "status": "success"
            })
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Productivity prediction failed",
                "error": str(e),
                 "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    @app.route('/generate-team-heatmap')
    def generate_team_heatmap():
        try:
            import os
        
            # ✅ Delete old model
            model_path = 'data/productivity_heatmap_ai_model.pkl'
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # ✅ Generate proper training data
            sample_training = []
            for day in range(5):  # Mon-Fri
                for hour in range(9, 18):  # 9 AM - 5 PM
                    # Morning hours (9-11) are most productive
                    if hour >= 9 and hour <= 11:
                        productivity = 0.75 + (hour - 9) * 0.05
                    # Afternoon dip (12-14)
                    elif hour >= 12 and hour <= 14:
                        productivity = 0.55 + (hour - 12) * 0.05
                    # Late afternoon recovery (15-17)
                    else:
                        productivity = 0.65 + (hour - 15) * 0.03
                
                    sample_training.append({
                        'user_id': f'user_{day}',
                        'hour_of_day': hour,
                        'day_of_week': day,
                        'meeting_count': 1 if hour in [10, 14] else 0,
                        'focus_events': 2 if hour in [9, 10, 15, 16] else 1,
                        'calendar_density': 0.5 + (hour % 3) * 0.1,
                        'message_activity': 20 + hour * 2,
                        'commit_activity': 3 if hour < 12 else 2,
                        'productivity_score': productivity
                    })
            heatmap_ai = ProductivityHeatmapAI()
            # Train with sample data
            training_result = heatmap_ai.train(sample_training)
        
            # Generate heatmap
            heatmap_data = heatmap_ai.get_team_heatmap(days_back=7)
            
            
            
            return jsonify({
                "message": "Team productivity heatmap generated!",
                "training_result": training_result,
                "heatmap_data": heatmap_data,
                "status": "success"
            })
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Heatmap generation failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    @app.route('/generate-team-heatmap-real')
    def generate_team_heatmap_real():
        """Generate heatmap from REAL calendar + GitHub data"""
        try:
            from models import CalendarData, CodeActivity
            from sqlalchemy import func
            from datetime import datetime, timedelta
        
            # Get real data from last 7 days
            seven_days_ago = datetime.now() - timedelta(days=7)
        
            # Initialize 5x9 matrix (5 days, 9 hours: 9AM-5PM)
            productivity_matrix = [[0 for _ in range(9)] for _ in range(5)]
        
            # Get calendar events grouped by day/hour
            calendar_events = CalendarData.query.filter(
                CalendarData.data_source == 'real',
                CalendarData.start_time >= seven_days_ago
            ).all()
        
            # Get code activity
            code_activities = CodeActivity.query.filter(
                CodeActivity.data_source == 'real',
                CodeActivity.date >= seven_days_ago.date()
            ).all()
        
            # Calculate productivity for each time slot
            for day in range(5):  # Mon-Fri
                for hour_idx in range(9):  # 9AM-5PM
                    hour = 9 + hour_idx

                    # Count meetings in this slot
                    meeting_count = sum(1 for event in calendar_events 
                        if event.start_time.weekday() == day and 
                            event.start_time.hour == hour)
                
                    # Count commits in this time period
                    commit_count = sum(activity.commits_count for activity in code_activities
                        if activity.date.weekday() == day)
                
                    # Calculate productivity score
                    # High commits + low meetings = high productivity
                    productivity = 0.5  # Base
                    productivity += min(0.3, commit_count * 0.05)  # Up to +0.3 for commits
                    productivity -= min(0.2, meeting_count * 0.1)  # Down to -0.2 for meetings
                
                    # Morning boost (9-11 AM most productive)
                    if hour_idx <= 2:
                        productivity += 0.2
                
                    productivity_matrix[day][hour_idx] = max(0.2, min(1.0, productivity))
        
            return jsonify({
                "status": "success",
                "heatmap_data": {
                    "heatmap_matrix": productivity_matrix,
                    "data_source": "real_calendar_and_github",
                    "events_analyzed": len(calendar_events),
                    "commits_analyzed": sum(a.commits_count for a in code_activities)
                }
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    @app.route('/api/member-heatmap/<username>')
    def get_member_heatmap(username):
        """Generate productivity heatmap for specific member from REAL data"""
        try:
            from models import CalendarData, CodeActivity
            from datetime import datetime, timedelta
        
            seven_days_ago = datetime.now() - timedelta(days=750)

            print(f"\n{'='*60}")
            print(f"📊 FETCHING HEATMAP FOR: {username}")
            print(f"{'='*60}")
            
            # Normalize username variations
            username_variants = [
                username,
                username.lower(),
                username.upper(),
                username.replace('-', ''),
                username.replace('_', ''),
                username.capitalize(),
                # For praneetaad078, also try praneetaAD078
                username.replace('ad', 'AD') if 'ad' in username.lower() else username,
                username.replace('AD', 'ad') if 'AD' in username else username
            ]
            print(f"Trying username variants: {username_variants}")
        
            # Get member's calendar events
            calendar_events = CalendarData.query.filter(
                CalendarData.data_source == 'real',
                CalendarData.user_id.in_(username_variants)
             ).filter(
                CalendarData.start_time >= seven_days_ago
            ).all()
            print(f"📅 Found {len(calendar_events)} calendar events")
            
            # Get member's code activity
            code_activities = CodeActivity.query.filter(
                CodeActivity.data_source == 'real',
                CodeActivity.user_id.in_(username_variants)
            ).filter(
                CodeActivity.date >= seven_days_ago.date()
            ).all()

            print(f"💻 Found {len(code_activities)} code activities")
            
            # ✅ If no data found, try checking database for ALL usernames
            if len(calendar_events) == 0 and len(code_activities) == 0:
                all_cal_users = db.session.query(CalendarData.user_id).filter(
                    CalendarData.data_source == 'real'
                ).distinct().all()
            
                all_code_users = db.session.query(CodeActivity.user_id).filter(
                    CodeActivity.data_source == 'real'
                ).distinct().all()
            
                print(f"⚠️ No data found for {username}")
                print(f"Available calendar users: {[u[0] for u in all_cal_users]}")
                print(f"Available code users: {[u[0] for u in all_code_users]}")
            
            # Initialize 5x9 matrix
            productivity_matrix = [[0.5 for _ in range(9)] for _ in range(5)]
        
            # Calculate productivity for each time slot
            for day in range(5):
                for hour_idx in range(9):
                    hour = 9 + hour_idx
                
                    # Count meetings
                    meeting_count = sum(1 for event in calendar_events 
                        if event.start_time.weekday() == day and 
                            event.start_time.hour == hour)
                
                    # Count commits for this day
                    day_commits = sum(activity.commits_count for activity in code_activities
                        if activity.date.weekday() == day)
                
                    # Calculate productivity
                    productivity = 0.5  # Base
                
                    # More commits = higher productivity
                    productivity += min(0.3, day_commits * 0.03)
                
                    # More meetings = lower productivity
                    productivity -= min(0.25, meeting_count * 0.15)
                
                    # Time-of-day adjustments
                    if hour_idx <= 2:  # 9-11 AM
                        productivity += 0.15
                    elif hour_idx >= 6:  # 3-5 PM
                        productivity -= 0.1
                
                    productivity_matrix[day][hour_idx] = max(0.2, min(1.0, productivity))

            total_commits = sum(a.commits_count for a in code_activities)
        
            print(f"✅ Heatmap generated: {len(calendar_events)} events, {total_commits} commits")
            print(f"{'='*60}\n")
            
            return jsonify({
                "status": "success",
                "username": username,
                "heatmap_matrix": productivity_matrix,
                "data_points": {
                    "calendar_events": len(calendar_events),
                    "code_activities": len(code_activities),
                    "total_commits": sum(a.commits_count for a in code_activities)
                }
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    # ==================== DEPENDENCY TRACKER (Part 5C) ====================
    
    @app.route('/api/team-productivity-summary/<int:project_id>')
    def get_team_productivity_summary(project_id):
        """Get comprehensive team productivity for a specific project"""
        try:
            from sqlalchemy import func, case
        
            # Get all project members (excluding manager)
            project_members = db.session.query(ProjectMember.username).filter(
                ProjectMember.project_id == project_id,
                ProjectMember.role != 'lead'
            ).all()
        
            member_usernames = [m[0] for m in project_members]
        
            if not member_usernames:
                return jsonify({
                    "status": "success",
                    "team_productivity": 0,
                    "total_tasks": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "pending": 0,
                    "members": []
                })
        
            # Get ALL tasks for this project
            all_tasks = Task.query.filter(
                Task.project_id == project_id,
                Task.assigned_to.in_(member_usernames)
            ).all()
        
            # Calculate overall statistics
            total_tasks = len(all_tasks)
            completed = len([t for t in all_tasks if t.status == 'completed'])
            in_progress = len([t for t in all_tasks if t.status == 'in_progress'])
            pending = len([t for t in all_tasks if t.status == 'pending'])
        
            # Calculate productivity for each member
            member_productivity = []
            total_productivity = 0
            active_members = 0
        
            for username in member_usernames:
                member_tasks = [t for t in all_tasks if t.assigned_to == username]
            
                if len(member_tasks) > 0:
                    member_completed = len([t for t in member_tasks if t.status == 'completed'])
                    member_in_progress = len([t for t in member_tasks if t.status == 'in_progress'])
                    member_pending = len([t for t in member_tasks if t.status == 'pending'])
                
                    # Calculate individual productivity
                    completion_rate = (member_completed / len(member_tasks)) * 100
                    activity_factor = min(100, (member_in_progress + member_completed) * 20)
                    productivity = (completion_rate * 0.7) + (activity_factor * 0.3)
                
                    total_productivity += productivity
                    active_members += 1
                
                    member_productivity.append({
                        "username": username,
                        "productivity": round(productivity, 1),
                        "total_tasks": len(member_tasks),
                        "completed": member_completed,
                        "in_progress": member_in_progress,
                        "pending": member_pending
                    })
                else:
                    # Member with no tasks - use baseline
                    member_productivity.append({
                        "username": username,
                        "productivity": 60,
                        "total_tasks": 0,
                        "completed": 0,
                        "in_progress": 0,
                        "pending": 0
                    })
        
            # Calculate team average
            team_productivity = round(total_productivity / active_members, 1) if active_members > 0 else 0
        
            return jsonify({
                "status": "success",
                "team_productivity": team_productivity,
                "total_tasks": total_tasks,
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending,
                "completion_rate": round((completed / total_tasks * 100) if total_tasks > 0 else 0, 1),
                "members": member_productivity,
                "active_members": active_members
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    @app.route('/test-dependency-model')
    def test_dependency_model():
        try:
            import os
            # ✅ Delete old model file
            model_path = 'data/dependency_tracker_model.pkl'
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"✅ Deleted old model: {model_path}")
        
            all_data = [
                {
                    'source_story': 'USER_STORY_001',
                    'dependent_story': 'USER_STORY_002',
                    'dependency_type': 'blocks',
                    'risk_probability': 0.7,
                    'cascade_impact': 4
                },
                {
                    'source_story': 'USER_STORY_002',
                    'dependent_story': 'USER_STORY_003',
                    'dependency_type': 'blocks',
                    'risk_probability': 0.4,
                    'cascade_impact': 2
                },
                {
                    'source_story': 'USER_STORY_001',
                    'dependent_story': 'USER_STORY_004',
                    'dependency_type': 'relates',
                    'risk_probability': 0.3,
                    'cascade_impact': 1
                },
                {
                    'source_story': 'TEST_STORY_A',
                    'dependent_story': 'TEST_STORY_B',
                    'dependency_type': 'blocks',
                    'risk_probability': 0.6,
                    'cascade_impact': 3
                }
            ]
            # ✅ Create NEW instance (won't load old model since we deleted it)
            dependency_tracker = DependencyTracker(load_saved_model=False)
             # Train fresh
            training_result = dependency_tracker.train(all_data)
            print(f"Training completed: {training_result}")
        
            
            # Predict on same data (this will work because predict() now handles untrained models)
            predictions = dependency_tracker.predict(all_data)
            visualization = dependency_tracker.get_dependency_visualization()
            
            return jsonify({
                "message": "Dependency Tracker test completed!",
                "training_result": training_result,
                "test_predictions": predictions,
                "dependency_visualization": visualization,
                "status": "success"
            })
            
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Dependency model test failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500
    
    @app.route('/analyze-dependencies')
    def analyze_dependencies():
        try:
            import os
            print("\n" + "="*60)
            print("🔗 ANALYZING DEPENDENCIES - BACKEND")
            print("="*60)
            # ✅ Delete old model
            model_path = 'data/dependency_tracker_model.pkl'
            if os.path.exists(model_path):
                os.remove(model_path)
            print(f"✅ Deleted old dependency model")
           
            
            try:
                coordinator = RealAPICoordinator()
                real_dependencies = coordinator.extract_github_dependencies(
                    github_usernames=['Rooba8925', 'praneetaAD078', 'sadhana-095'],
                    days_back=730
                )
                print(f"✅ Found {len(real_dependencies)} real GitHub dependencies")
            except Exception as github_error:
                print(f"⚠️ GitHub dependency extraction failed: {str(github_error)}")
                real_dependencies = []
        
        
            #If no real dependencies found, use sample data
            if len(real_dependencies) == 0:
                print("⚠️ No real dependencies found, using sample data")
                real_dependencies = [
                    {
                        'source_story': 'rooba8925/Authentication_Module',
                        'dependent_story': 'praneetaad078/User_Dashboard',
                        'source_task_name': 'Authentication Module',
                        'dependent_task_name': 'User Dashboard',
                        'dependency_type': 'blocks',
                        'risk_probability': 0.6,
                        'cascade_impact': 3
                    },
                    {
                        'source_story': 'praneetaad078/API_Integration',
                        'dependent_story': 'sadhana-095/Data_Visualization',
                        'source_task_name': 'API Integration',
                        'dependent_task_name': 'Data Visualization',
                        'dependency_type': 'blocks',
                        'risk_probability': 0.5,
                        'cascade_impact': 2
                    },
                    {
                        'source_story': 'sadhana-095/Database_Schema',
                        'dependent_story': 'rooba8925/Backend_API',
                        'source_task_name': 'Database Schema',
                        'dependent_task_name': 'Backend API',
                        'dependency_type': 'blocks',
                        'risk_probability': 0.4,
                        'cascade_impact': 2
                    }
                ]
        
            
            dependency_tracker = DependencyTracker(load_saved_model=False)
            # ✅ FIX: Train the model first before predicting
            training_result = dependency_tracker.train(real_dependencies)
            print(f"✅ Training complete: {training_result}")
            # Now predict with the trained model
            predictions = dependency_tracker.predict(real_dependencies)
            print(f"✅ Generated {len(predictions.get('predictions', []))} predictions")
        
            # ✅ Get visualization
            visualization = dependency_tracker.get_dependency_visualization()
            print("="*60)
            print("✅ DEPENDENCY ANALYSIS COMPLETE")
            print("="*60 + "\n")
            return jsonify({
                "message": "Dependency analysis completed!",
                "status": "success",
                "dependency_predictions": predictions,
                "training_result": training_result,  # ✅ Add training info
                "dependency_predictions": predictions,
                "network_visualization": visualization,
                "data_source": "real" if len(real_dependencies) > 3 else "sample",
                "total_dependencies": len(real_dependencies)

            })
            
        except Exception as e:
            import traceback
            print("\n" + "="*60)
            print("❌ DEPENDENCY ANALYSIS FAILED")
            print("="*60)
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print("="*60 + "\n")
            return jsonify({
                "message": "Dependency analysis failed",
                "error": str(e),
                "traceback": traceback.format_exc(),  # ✅ Better error info
                "status": "error"
            }), 500

    @app.route('/api/member-dependencies/<username>')
    def get_member_dependencies(username):
        """Get dependencies specific to a member"""
        try:
            print(f"\n{'='*60}")
            print(f"🔗 FETCHING DEPENDENCIES FOR: {username}")
            print(f"{'='*60}")
            
            # Get real dependencies from GitHub
            coordinator = RealAPICoordinator()
        
            # ✅ FIX: Use correct GitHub usernames with capital letters
            github_users = ['Rooba8925', 'praneetaAD078', 'sadhana-095']
        
            print(f"Checking dependencies for GitHub users: {github_users}")

            all_dependencies = coordinator.extract_github_dependencies(
                github_usernames=github_users,
                days_back=750
            )
        
            print(f"Total dependencies found: {len(all_dependencies)}")
        
            if len(all_dependencies) == 0:
                return jsonify({
                    "status": "success",
                    "dependencies": [],
                    "count": 0,
                    "message": "No GitHub dependencies found. Make sure you have issues with keywords like 'depends on', 'blocks', etc."
                })
        
            # Filter dependencies for this user
            user_dependencies = []
            username_lower = username.lower()
        
            for dep in all_dependencies:
                source = (dep.get('source_story', '')).lower()
                dependent = (dep.get('dependent_story', '')).lower()
            
                # Check if user is involved
                if username_lower in source or username_lower in dependent:
                    user_dependencies.append(dep)
                    print(f"  ✅ Match: {dep.get('source_task_name')} → {dep.get('dependent_task_name')}")
        
            print(f"Dependencies for {username}: {len(user_dependencies)}")
        
            # Train model if we have dependencies
            if len(all_dependencies) > 0:
                import os
                model_path = 'data/dependency_tracker_model.pkl'
                if os.path.exists(model_path):
                    os.remove(model_path)
            
                dependency_tracker = DependencyTracker(load_saved_model=False)
                training_result = dependency_tracker.train(all_dependencies)
            
                # Predict on user's dependencies or all if none found
                predict_data = user_dependencies if user_dependencies else all_dependencies[:5]
                predictions = dependency_tracker.predict(predict_data)
            
                filtered_predictions = predictions.get('predictions', [])
            
                return jsonify({
                    "status": "success",
                    "dependencies": filtered_predictions,
                    "count": len(filtered_predictions),
                    "training_result": training_result,
                    "all_dependencies_count": len(all_dependencies)
                })
        
            return jsonify({
                "status": "success",
                "dependencies": [],
                "count": 0
            })
        
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR in get_member_dependencies:")
            print(traceback.format_exc())
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    @app.route('/create-test-github-issues')
    def create_test_github_issues():
        """Create test GitHub issues with dependencies"""
        try:
            coordinator = RealAPICoordinator()
        
            if not coordinator.github_extractor:
                return jsonify({
                    "status": "error",
                    "error": "GitHub not configured"
                }), 400
        
            # Detect GitHub client
            github_client = None
            if hasattr(coordinator.github_extractor, 'client'):
                github_client = coordinator.github_extractor.client
            elif hasattr(coordinator.github_extractor, 'github'):
                github_client = coordinator.github_extractor.github
            else:
                return jsonify({
                    "status": "error",
                    "error": "Cannot find GitHub client"
                }), 500
        
            # Get authenticated user
            user = github_client.get_user()
        
            # Get first repo (or specify your repo)
            repos = list(user.get_repos())
            if len(repos) == 0:
                return jsonify({
                    "status": "error",
                    "error": "No repositories found"
                }), 404
        
            repo = repos[0]  # Use first repo
        
            print(f"Creating test issues in: {repo.full_name}")
        
            # Create test issues
            issue1 = repo.create_issue(
                title="Setup Database Schema",
                body="This issue must be completed first.\nBlocks: User authentication and dashboard features."
            )
        
            issue2 = repo.create_issue(
                title="Implement User Authentication",
                body=f"This feature depends on #{issue1.number} (Database Setup)\nBlocked by: Database schema completion"
            )
        
            issue3 = repo.create_issue(
                title="Build User Dashboard",
                body=f"Waiting for #{issue2.number} (User Authentication)\nThis requires authentication to be completed first.\nAlso depends on #{issue1.number}"
            )
        
            return jsonify({
                "status": "success",
                "message": f"Created 3 test issues in {repo.full_name}",
                "issues": [
                    {"number": issue1.number, "title": issue1.title, "url": issue1.html_url},
                    {"number": issue2.number, "title": issue2.title, "url": issue2.html_url},
                    {"number": issue3.number, "title": issue3.title, "url": issue3.html_url}
                ]
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
            
    # ==================== TIME-SERIES & CALENDAR DENSITY ENDPOINTS ====================

    @app.route('/time-series-analysis')
    def time_series_analysis():
        """Run time-series aggregation analysis"""
        try:
            # Get processed features first
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
        
            sprint_features = processing_results.get('sample_sprint_features', [])
            productivity_features = processing_results.get('sample_productivity_features', [])
        
            # Run time-series aggregation
            aggregator = TimeSeriesAggregator()
        
            sprint_trends = aggregator.aggregate_sprint_metrics(sprint_features)
            productivity_patterns = aggregator.aggregate_productivity_patterns(productivity_features)
            velocity_trends = aggregator.calculate_velocity_trends(sprint_features, window_size=3)
        
            return jsonify({
                "message": "Time-series analysis completed successfully!",
                "sprint_trends": sprint_trends,
                "productivity_patterns": productivity_patterns,
                "velocity_trends": velocity_trends,
                "analysis_period": "7 days",
                "total_data_points": len(sprint_features),
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Time-series analysis failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500
    
    @app.route('/calendar-density-analysis')
    def calendar_density_analysis():
        """Run calendar density analysis"""
        try:
            # Get processed features
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
        
            sprint_features = processing_results.get('sample_sprint_features', [])
        
            if not sprint_features:
                return jsonify({
                    "message": "No calendar data available for analysis",
                    "status": "error"
                }), 404
        
            # Run calendar density analysis
            analyzer = CalendarDensityAnalyzer()
            density_analysis = analyzer.analyze_team_density(sprint_features)
        
            return jsonify({
                "message": "Calendar density analysis completed!",
                "analysis": density_analysis,
                "team_members_analyzed": len(set(f['user_id'] for f in sprint_features)),
                "analysis_period": "7 days",
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Calendar density analysis failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500


    @app.route('/combined-analysis')
    def combined_analysis():
        """Run both time-series and calendar density analysis"""
        try:
            # Get processed features
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
        
            sprint_features = processing_results.get('sample_sprint_features', [])
            productivity_features = processing_results.get('sample_productivity_features', [])
        
            # Time-series analysis
            aggregator = TimeSeriesAggregator()
            sprint_trends = aggregator.aggregate_sprint_metrics(sprint_features)
            productivity_patterns = aggregator.aggregate_productivity_patterns(productivity_features)
            velocity_trends = aggregator.calculate_velocity_trends(sprint_features)
        
            # Calendar density analysis
            analyzer = CalendarDensityAnalyzer()
            density_analysis = analyzer.analyze_team_density(sprint_features)
        
            return jsonify({
                "message": "Combined analysis completed!",
                "time_series": {
                    "sprint_trends": sprint_trends,
                    "productivity_patterns": productivity_patterns,
                    "velocity_trends": velocity_trends
                },
                "calendar_density": density_analysis,
                "summary": {
                    "total_team_members": len(set(f['user_id'] for f in sprint_features)),
                    "analysis_period": "7 days",
                    "data_points_analyzed": len(sprint_features),
                    "avg_team_density": density_analysis.get('overall_stats', {}).get('avg_calendar_density', 0),
                    "high_risk_users": len(density_analysis.get('risk_indicators', {}).get('high_risk_users', []))
                },
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Combined analysis failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    @app.route('/debug-users')
    def debug_users():
        from models import CodeActivity, CommunicationData, CalendarData
    
        github_users = CodeActivity.query.with_entities(CodeActivity.user_id).distinct().all()
        slack_users = CommunicationData.query.with_entities(CommunicationData.user_id).distinct().all()
        calendar_users = CalendarData.query.with_entities(CalendarData.user_id).distinct().all()
    
        return jsonify({
            "github_users": [u[0] for u in github_users if u[0]],
            "slack_users": [u[0] for u in slack_users if u[0]],
            "calendar_users": [u[0] for u in calendar_users if u[0]],
            "status": "success"
        })

    @app.route('/check-table-columns')
    def check_table_columns():
        """Check actual column names in tables"""
        try:
            from sqlalchemy import text, inspect
        
            tables_to_check = [
                'calendar_data',
                'communication_data', 
                'code_activity',
                'sprint_health_metrics',
                'productivity_patterns'
            ]
        
            table_info = {}
        
            for table_name in tables_to_check:
                try:
                    # Get column info
                    inspector = inspect(db.engine)
                    columns = inspector.get_columns(table_name)
                
                    # Find datetime/timestamp columns
                    time_columns = [
                        col['name'] for col in columns 
                        if 'time' in col['name'].lower() or 
                            'date' in col['name'].lower() or
                            'created' in col['name'].lower() or
                            str(col['type']).lower() in ['timestamp', 'datetime', 'date']
                    ]
                
                    table_info[table_name] = {
                        'exists': True,
                        'all_columns': [col['name'] for col in columns],
                        'time_columns': time_columns,
                        'column_types': {col['name']: str(col['type']) for col in columns}
                    }
                
                except Exception as e:
                    table_info[table_name] = {
                        'exists': False,
                        'error': str(e)
                    }
        
            return jsonify({
                "message": "Table column information",
                "tables": table_info,
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Column check failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500


    # ==================== REDIS CACHE TEST ENDPOINT ====================

    @app.route('/cache-test')
    def cache_test():
        """Test Redis cache functionality"""
        try:
        
            cache = RedisCache()
        
            # Test data
            test_data = {
                'test': 'value',
                'timestamp': datetime.utcnow().isoformat()
            }
        
            # Try to set cache
            set_result = cache.set_cache('test_key', test_data, 60)
        
            # Try to get cache
            retrieved_data = cache.get_cache('test_key')
        
            return jsonify({
                "message": "Redis cache test",
                "redis_connected": cache.redis_client is not None,
                "cache_working": set_result and retrieved_data is not None,
                "test_data_set": test_data,
                "test_data_retrieved": retrieved_data,
                "status": "success" if cache.redis_client else "redis_not_available"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Cache test failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    #-----------------------------------real team member------------------------------#
    @app.route('/api/real-team-members')
    def get_real_team_members():
        """Get list of users who have real data in the system"""
        try:
            from models import CodeActivity, CalendarData, CommunicationData
            from sqlalchemy import distinct
        
            # Get users with real GitHub activity
            github_users = db.session.query(distinct(CodeActivity.user_id)).filter(
                CodeActivity.data_source == 'real'
            ).all()
        
            # Get users with real calendar data
            calendar_users = db.session.query(distinct(CalendarData.user_id)).filter(
                CalendarData.data_source == 'real'
            ).all()
        
            # Combine and deduplicate
            all_users = set([u[0] for u in github_users] + [u[0] for u in calendar_users])
        
            return jsonify({
                "status": "success",
                "real_users": list(all_users),
                "count": len(all_users)
            })
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500

    # ==================== TIMESCALEDB SETUP ====================
    @app.route('/setup-timescaledb-correct')
    def setup_timescaledb_correct():
        """Setup TimescaleDB with CORRECT column names from your schema"""
        try:
            from sqlalchemy import text
        
            # ✅ CORRECT column names based on your actual schema
            hypertables = [
                {
                    'table': 'calendar_data',
                    'time_column': 'start_time',  # ✅ Correct: start_time (not event_start)
                    'chunk_interval': '1 week'
                },
                {
                    'table': 'communication_data',
                    'time_column': 'date',  # ✅ Correct: date
                    'chunk_interval': '1 week'
                },
                {
                    'table': 'code_activity',
                    'time_column': 'date',  # ✅ Correct: date (not activity_date)
                    'chunk_interval': '1 week'
                },
                {
                    'table': 'sprint_health_metrics',
                    'time_column': 'created_at',  # ✅ Correct: created_at
                    'chunk_interval': '1 day'
                },
                {
                    'table': 'productivity_patterns',
                    'time_column': 'time_slot',  # ✅ Better: time_slot (more specific than created_at)
                    'chunk_interval': '1 day'
                }
            ]
        
            results = []
        
            for ht in hypertables:
                try:
                    with db.engine.begin() as conn:
                        # Check if table has data
                        count_query = text(f"SELECT COUNT(*) FROM {ht['table']};")
                        count = conn.execute(count_query).scalar()
                    
                        # Create hypertable
                        create_query = text(f"""
                            SELECT create_hypertable(
                                '{ht['table']}', 
                                '{ht['time_column']}',
                                chunk_time_interval => INTERVAL '{ht['chunk_interval']}',
                                if_not_exists => TRUE,
                                migrate_data => TRUE
                            );
                        """)
                    
                        result = conn.execute(create_query)
                    
                        results.append({
                            'table': ht['table'],
                            'time_column': ht['time_column'],
                            'chunk_interval': ht['chunk_interval'],
                            'rows_migrated': count,
                            'status': 'created'
                        })
                    
                except Exception as e:
                    error_msg = str(e).lower()
                
                    if 'already a hypertable' in error_msg:
                        results.append({
                            'table': ht['table'],
                            'status': 'already_hypertable'
                        })
                    elif 'table_name must be a valid relation' in error_msg:
                        results.append({
                            'table': ht['table'],
                            'status': 'table_not_found'
                        })
                    else:
                        results.append({
                            'table': ht['table'],
                            'status': 'failed',
                            'error': str(e)[:200]
                        })
        
            # Verify created hypertables
            with db.engine.connect() as conn:
                verify_query = text("""
                    SELECT 
                        hypertable_name,
                        num_chunks,
                        num_dimensions,
                        compression_enabled
                    FROM timescaledb_information.hypertables
                    ORDER BY hypertable_name;
                """)
                hypertables_list = [dict(row._mapping) for row in conn.execute(verify_query).fetchall()]
        
            return jsonify({
                "message": "TimescaleDB setup completed with correct column names!",
                "hypertables_created": results,
                "total_hypertables": len(hypertables_list),
                "active_hypertables": hypertables_list,
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "TimescaleDB setup failed",
                "error": str(e),
                "traceback": traceback.format_exc()[:1000],
                "status": "error"
            }), 500


    @app.route('/enable-compression')
    def enable_compression():
        """Enable TimescaleDB compression for better performance"""
        try:
            from sqlalchemy import text
        
            hypertables = [
                'calendar_data',
                'communication_data',
                'code_activity',
                'sprint_health_metrics',
                'productivity_patterns'
            ]
        
            results = []
        
            for table in hypertables:
                try:
                    with db.engine.begin() as conn:
                        # Add compression policy (compress data older than 7 days)
                        compress_query = text(f"""
                            SELECT add_compression_policy('{table}', INTERVAL '7 days');
                        """)
                    
                        conn.execute(compress_query)
                    
                        results.append({
                            'table': table,
                            'compression': 'enabled',
                            'policy': 'compress_after_7_days'
                        })
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'already exists' in error_msg or 'compression already' in error_msg:
                        results.append({
                            'table': table,
                            'compression': 'already_enabled'
                        })
                    else:
                        results.append({
                            'table': table,
                            'compression': 'failed',
                            'error': str(e)[:150]
                        })
        
            return jsonify({
                "message": "Compression policies configured",
                "results": results,
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "message": "Compression setup failed",
                "error": str(e),
                "traceback": traceback.format_exc()[:1000],
                "status": "error"
            }), 500



    @app.route('/user-data-summary')
    def user_data_summary():
        from models import CodeActivity, CommunicationData, CalendarData
        from processing.user_mapping import normalize_user_id
    
        users_summary = {}
    
        # Get all real users
        real_users = {'praneetaad078', 'rooba8925', 'sadhana-095'}
    
        for user in real_users:
            users_summary[user] = {
                'github_commits': CodeActivity.query.filter_by(data_source='real').filter(
                    CodeActivity.user_id.in_(['praneetaAD078', 'Rooba8925', 'sadhana-095'])
                ).count(),
                'calendar_events': CalendarData.query.filter_by(data_source='real').count(),
                'slack_messages': CommunicationData.query.filter_by(data_source='real').count()
            }
    
        return jsonify({
            "users": users_summary,
            "status": "success"
        })

    @app.route('/verify-real-data')
    def verify_real_data():
        """Verify what data sources are real vs mock"""
        try:
            from models import CalendarData, CommunicationData, CodeActivity
            from sqlalchemy import func
        
            # Calendar breakdown
            calendar_stats = db.session.query(
                CalendarData.data_source,
                func.count(CalendarData.id).label('count'),
                func.min(CalendarData.created_at).label('first_date'),
                func.max(CalendarData.created_at).label('last_date')
            ).group_by(CalendarData.data_source).all()
        
            # Communication breakdown
            comm_stats = db.session.query(
                CommunicationData.data_source,
                func.count(CommunicationData.id).label('count'),
                func.min(CommunicationData.created_at).label('first_date'),
                func.max(CommunicationData.created_at).label('last_date')
            ).group_by(CommunicationData.data_source).all()
        
            # Code activity breakdown  
            code_stats = db.session.query(
                CodeActivity.data_source,
                func.count(CodeActivity.id).label('count'),
                func.min(CodeActivity.created_at).label('first_date'),
                func.max(CodeActivity.created_at).label('last_date')
            ).group_by(CodeActivity.data_source).all()
        
            # Get sample records to verify
            latest_calendar = CalendarData.query.filter_by(data_source='real').order_by(CalendarData.created_at.desc()).limit(3).all()
            latest_comm = CommunicationData.query.filter_by(data_source='real').order_by(CommunicationData.created_at.desc()).limit(3).all()
            latest_code = CodeActivity.query.filter_by(data_source='real').order_by(CodeActivity.created_at.desc()).limit(3).all()
        
            return jsonify({
                "message": "Real data verification",
                "calendar": {
                    "breakdown": [{"source": s[0], "count": s[1], "first_date": str(s[2]), "last_date": str(s[3])} for s in calendar_stats],
                    "latest_real_samples": [{"title": c.title, "start": str(c.start_time), "user": c.user_id} for c in latest_calendar]
                },
                "communication": {
                    "breakdown": [{"source": s[0], "count": s[1], "first_date": str(s[2]), "last_date": str(s[3])} for s in comm_stats],
                    "latest_real_samples": [{"messages": c.message_count, "date": str(c.date), "user": c.user_id} for c in latest_comm]
                },
                "code_activity": {
                    "breakdown": [{"source": s[0], "count": s[1], "first_date": str(s[2]), "last_date": str(s[3])} for s in code_stats],
                    "latest_real_samples": [{"commits": c.commits_count, "date": str(c.date), "user": c.user_id, "repo": c.repository} for c in latest_code]
                },
                "verification_status": {
                    "has_real_calendar": any(s[0] == 'real' for s in calendar_stats),
                    "has_real_communication": any(s[0] == 'real' for s in comm_stats),
                    "has_real_code": any(s[0] == 'real' for s in code_stats)
                },
                "status": "success"
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }), 500

    @app.route('/data-summary-fast')
    def data_summary_fast():
        """Quick data summary without re-running ETL"""
        try:
            from models import CalendarData, CommunicationData, CodeActivity
            from sqlalchemy import func
        
            # Get counts by source
            calendar_real = CalendarData.query.filter_by(data_source='real').count()
            calendar_mock = CalendarData.query.filter_by(data_source='mock').count()
            calendar_total = CalendarData.query.count()
        
            comm_real = CommunicationData.query.filter_by(data_source='real').count()
            comm_mock = CommunicationData.query.filter_by(data_source='mock').count()
            comm_total = CommunicationData.query.count()
        
            code_real = CodeActivity.query.filter_by(data_source='real').count()
            code_mock = CodeActivity.query.filter_by(data_source='mock').count()
            code_total = CodeActivity.query.count()
        
            # Calculate percentages
            cal_pct = round((calendar_real / calendar_total * 100) if calendar_total > 0 else 0, 1)
            comm_pct = round((comm_real / comm_total * 100) if comm_total > 0 else 0, 1)
            code_pct = round((code_real / code_total * 100) if code_total > 0 else 0, 1)
        
            return jsonify({
                "message": "Data summary (instant)",
                "calendar": {
                    "real": calendar_real,
                    "mock": calendar_mock,
                    "total": calendar_total,
                    "real_percentage": f"{cal_pct}%"
                },
                "communication": {
                    "real": comm_real,
                    "mock": comm_mock,
                    "total": comm_total,
                    "real_percentage": f"{comm_pct}%"
                },
                "code_activity": {
                    "real": code_real,
                    "mock": code_mock,
                    "total": code_total,
                    "real_percentage": f"{code_pct}%"
                },
                "overall_real_percentage": round((calendar_real + comm_real + code_real) / (calendar_total + comm_total + code_total) * 100, 1) if (calendar_total + comm_total + code_total) > 0 else 0,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/dependency-explanation/<story_id>')
    def dependency_explanation(story_id):
        """Explain why a dependency exists and its impact"""
        try:
            # This would query your dependency data
            explanation = {
                "story_id": story_id,
                "why_dependent": "This story depends on authentication module completion because it requires user session management.",
                "blocking_reason": "Cannot implement user-specific features without auth foundation.",
                "cascade_impact": "Blocking 3 downstream stories: User Dashboard, Profile Management, Notifications",
                "mitigation_strategies": [
                    "Prioritize authentication story in current sprint",
                    "Consider parallel development with mocked auth",
                    "Split into smaller, independently deliverable pieces"
                ],
                "estimated_delay": "2-3 days if not resolved",
                "critical_path": True
            }
        
            return jsonify({
                "explanation": explanation,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500


    # ==================== TASK MANAGEMENT ====================

    @app.route('/api/tasks', methods=['POST'])
    def create_task():
        """Assign a task to a team member"""
        try:
            data = request.get_json()
            user = User.query.filter_by(username=data.get('assigned_by')).first()
        
            if not user or user.role != 'manager':
                return jsonify({
                    "error": "Only managers can assign tasks",
                    "status": "error"
                }), 403

            # Get project info if project_id provided
            project = None
            project_name = None
            if data.get('project_id'):
                project = db.session.get(Project, data['project_id'])
                project_name = project.name if project else None
            task = Task(
                project_id=data.get('project_id'),  # ✅ Include project_id
                title=data['title'],
                description=data.get('description'),
                assigned_to=data['assigned_to'],
                assigned_by=data['assigned_by'],
                priority=data.get('priority', 'medium'),
                due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
            )
        
            db.session.add(task)
            db.session.commit()
        
            # Send email notification
            assignee = User.query.filter_by(username=data['assigned_to']).first()
            if assignee:
                email_sent = send_task_assignment_email(assignee, task, project_name)
        
            return jsonify({
                "message": "Task assigned successfully",
                "task": task.to_dict(),
                "email_sent": email_sent,  # ✅ Include email status
                "status": "success"
            }), 201
        
        except Exception as e:
            db.session.rollback()
            print(f"❌ Task creation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/tasks/<int:task_id>/status', methods=['PUT'])
    def update_task_status(task_id):
        """Update task status (for members starting tasks)"""
        try:
            data = request.json
            new_status = data.get('status')
        
            if new_status not in ['pending', 'in_progress', 'completed']:
                return jsonify({'status': 'error', 'error': 'Invalid status'}), 400
        
            # ✅ Use SQLAlchemy instead of raw SQL
            task = Task.query.get(task_id)

            if not task:
                return jsonify({'status': 'error', 'error': 'Task not found'}), 404
        
            # Calculate completion percentage based on status
            completion_percentage = 0
            if new_status == 'in_progress':
                task.completion_percentage = 30
            elif new_status == 'completed':
                task.completion_percentage = 100
            else:
                task.completion_percentage = 0
            task.status = new_status
            task.updated_at = datetime.utcnow()
        
            db.session.commit()
           
            
        
            
        
            return jsonify({
                'status': 'success',
                'message': f'Task status updated to {new_status}',
                'task': task.to_dict()
            })
        
        except Exception as e:
            db.session.rollback()
            print(f"Error updating task status: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/api/tasks/<int:task_id>/submit', methods=['POST'])
    def submit_task(task_id):
        """Submit completed task with file and notify manager"""
        try:
            print(f"\n{'='*60}")
            print(f"📤 TASK SUBMISSION REQUEST")
            print(f"{'='*60}")
            print(f"Task ID: {task_id}")
        
            # ✅ Get JSON data (your frontend sends JSON, not form data)
            data = request.get_json()
            username = data.get('username')
            submission_notes = data.get('submission_notes')
            submission_file = data.get('submission_file')
            additional_links = data.get('additional_links')
        
            print(f"Submitted by: {username}")
            print(f"Notes: {submission_notes[:50]}..." if submission_notes and len(submission_notes) > 50 else f"Notes: {submission_notes}")
            print(f"File: {submission_file}")
            
            # ✅ Validate required fields
            if not username:
                print(f"❌ Username missing")
                return jsonify({
                    'status': 'error',
                    'error': 'Username is required'
                }), 400
        
            if not submission_notes:
                print(f"❌ Submission notes missing")
                return jsonify({
                    'status': 'error',
                    'error': 'Submission description is required'
                }), 400
            
            # Get the task
            task = Task.query.get(task_id)
        
            if not task:
                print(f"❌ Task {task_id} not found")
                return jsonify({
                    'status': 'error',
                    'error': 'Task not found'
                }), 404
            
            print(f"📋 Task found: '{task.title}'")
            print(f"👤 Task assigned to: '{task.assigned_to}'")
            print(f"🔍 Username comparison:")
            print(f"   - From request: '{username}' (type: {type(username)})")
            print(f"   - From task: '{task.assigned_to}' (type: {type(task.assigned_to)})")
            
            # ✅ Authorization check with flexible matching
            task_assigned_to = task.assigned_to.lower().strip() if task.assigned_to else ""
            submitting_user = username.lower().strip() if username else ""
        
            print(f"   - Lowercase comparison: '{submitting_user}' vs '{task_assigned_to}'")
            print(f"   - Are they equal? {submitting_user == task_assigned_to}")
            
            # ✅ Flexible matching (exact or partial)
            if task_assigned_to != submitting_user:
                # Try partial match
                if task_assigned_to not in submitting_user and submitting_user not in task_assigned_to:
                    print(f"❌ AUTHORIZATION FAILED")
                    print(f"   Expected: '{task.assigned_to}'")
                    print(f"   Got: '{username}'")
                    return jsonify({
                        'status': 'error',
                        'error': f'Unauthorized: This task is assigned to {task.assigned_to}, but you are {username}'
                    }), 403
        
            print(f"✅ Authorization successful")

            # ✅ Update task using SQLAlchemy
            task.status = 'completed'
            task.completion_percentage = 100
            task.submission_notes = submission_notes
            task.submission_file = submission_file
            task.additional_links = additional_links
            task.submitted_at = datetime.utcnow()
            task.updated_at = datetime.utcnow()
        
            db.session.commit()
            print(f"✅ Task marked as completed in database")
        
            # ✅ Send notification to manager
            manager = User.query.filter_by(username=task.assigned_by).first()
            if manager:
                notification = Notification(
                    recipient_username=manager.username,
                    title=f"Task Completed: {task.title}",
                    message=f"{username} has completed the task '{task.title}' and submitted their work.",
                    notification_type='task_completion',
                    is_read=False
                )
                db.session.add(notification)
                db.session.commit()
                print(f"✅ Notification sent to {manager.username}")
            
                # ✅ Send email to manager
                try:
                    send_task_completion_email(
                        manager=manager,
                        task=task,
                        submitted_by=username,
                        description=submission_notes,
                        file_name=submission_file,
                        links=additional_links
                    )
                    print(f"✅ Email sent to manager")
                except Exception as email_error:
                    print(f"⚠️ Email notification failed: {email_error}")
        
            print(f"{'='*60}\n")
        
            return jsonify({
                'status': 'success',
                'message': 'Task submitted successfully and manager notified',
                'task': task.to_dict()
            })
        
        except Exception as e:
            db.session.rollback()
            import traceback
            print(f"\n{'='*60}")
            print(f"❌ TASK SUBMISSION ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500

    @app.route('/notifications')
    def notifications_page():
        """Serve notifications page"""
        return send_file('../frontend/templates/notifications.html')

    # ==================== UTILITY FUNCTION TO ADD ====================

    def getReadableTaskName(story_id):
        """
        Helper function for frontend to convert story IDs to readable names
        ADD THIS to your app.py
        """
        # Remove repo name prefix if exists
        if '/' in story_id:
            story_id = story_id.split('/')[-1]
    
        # Remove issue number suffix  
        story_id = story_id.split('#')[0]
    
        # Convert snake_case or kebab-case to Title Case
        readable = story_id.replace('_', ' ').replace('-', ' ').title()
    
        return readable if readable else story_id
    
    @app.route('/api/tasks/<username>', methods=['GET'])
    def get_user_tasks(username):
        """Get all tasks for a user"""
        try:
            tasks = Task.query.filter_by(assigned_to=username).order_by(Task.created_at.desc()).all()
        
            # Calculate task statistics
            total = len(tasks)
            completed = len([t for t in tasks if t.status == 'completed'])
            in_progress = len([t for t in tasks if t.status == 'in_progress'])
            pending = len([t for t in tasks if t.status == 'pending'])
        
            return jsonify({
                "tasks": [t.to_dict() for t in tasks],
                "statistics": {
                    "total": total,
                    "completed": completed,
                    "in_progress": in_progress,
                    "pending": pending,
                    "completion_rate": round((completed / total * 100) if total > 0 else 0, 1)
                },
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/tasks/<int:task_id>', methods=['PUT'])
    def update_task(task_id):
        """Update task status and progress"""
        try:
            data = request.get_json()
            task = Task.query.get(task_id)
        
            if not task:
                return jsonify({
                    "error": "Task not found",
                    "status": "error"
                }), 404
        
            # Update fields
            if 'status' in data:
                task.status = data['status']
            if 'completion_percentage' in data:
                task.completion_percentage = data['completion_percentage']
            if 'priority' in data:
                task.priority = data['priority']
        
            task.updated_at = datetime.utcnow()
            db.session.commit()
        
            return jsonify({
                "message": "Task updated successfully",
                "task": task.to_dict(),
                "status": "success"
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
    def delete_task(task_id):
        """Delete a task"""
        try:
            task = Task.query.get(task_id)
        
            if not task:
                return jsonify({
                    "error": "Task not found",
                    "status": "error"
                }), 404
        
            db.session.delete(task)
            db.session.commit()
        
            return jsonify({
                "message": "Task deleted successfully",
                "status": "success"
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/tasks/all', methods=['GET'])
    def get_all_tasks():
        """Get all tasks (for manager view)"""
        try:
            tasks = Task.query.order_by(Task.created_at.desc()).all()
        
            return jsonify({
                "tasks": [t.to_dict() for t in tasks],
                "total": len(tasks),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500        
 #----------------------Team member deletion entirely----------------

    @app.route('/api/team-members/<username>', methods=['DELETE'])
    def delete_team_member(username):
        """Delete a team member"""
        try:
            member = User.query.filter_by(username=username, role='member').first()
        
            if not member:
                return jsonify({
                    "error": "Member not found or not a team member",
                    "status": "error"
                }), 404
        
            # Don't allow deleting managers
            if member.role == 'manager':
                return jsonify({
                    "error": "Cannot delete manager accounts",
                    "status": "error"
                }), 403
        
            db.session.delete(member)
            db.session.commit()
        
            return jsonify({
                "message": f"Member {username} deleted successfully",
                "status": "success"
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/team-members', methods=['GET'])
    def get_all_team_members():
        """Get all team members from database"""
        try:
            members = User.query.filter_by(role='member').all()
        
            return jsonify({
                "members": [m.to_dict() for m in members],
                "total": len(members),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    
    
    # ==================== PROJECT MANAGEMENT ====================

    @app.route('/api/projects', methods=['POST'])
    def create_project():
        """Create a new project (Manager only)"""
        try:
            data = request.get_json()
        
            project = Project(
                name=data['name'],
                description=data.get('description'),
                created_by=data['created_by']
            )
        
            db.session.add(project)
            db.session.commit()
        
            # Add creator as project lead
            project_member = ProjectMember(
                project_id=project.id,
                username=data['created_by'],
                role='lead'
            )
            db.session.add(project_member)
            db.session.commit()
        
            return jsonify({
                "message": "Project created successfully",
                "project": project.to_dict(),
                "status": "success"
            }), 201
        
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/projects', methods=['GET'])
    def get_projects():
        """Get projects for user"""
        try:
            username = request.args.get('username')
            user = User.query.filter_by(username=username).first()
        
            if user.role == 'manager':
                projects = Project.query.filter_by(created_by=username).all()
            else:
                project_members = ProjectMember.query.filter_by(username=username).all()
                project_ids = [pm.project_id for pm in project_members]
                projects = Project.query.filter(Project.id.in_(project_ids)).all() if project_ids else []
        
            return jsonify({
                "projects": [p.to_dict() for p in projects],
                "total": len(projects),
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500


    @app.route('/api/projects/<int:project_id>/members', methods=['POST'])
    def add_project_member(project_id):
        """Add member to project"""
        try:
            data = request.get_json()
            username = data.get('username')

            print(f"\n{'='*60}")
            print(f"📋 ADD MEMBER REQUEST")
            print(f"{'='*60}")
            print(f"Project ID: {project_id}")
            print(f"Username: {username}")
            
            # Check if user exists
            user = User.query.filter_by(username=username).first()
        
            if not user:
                print(f"❌ User '{username}' not found in database")
                return jsonify({'status': 'error', 'error': 'User not found'}), 404
            print(f"✅ User found: {user.full_name} ({user.email})")

            # Check if already a member
            existing_member = ProjectMember.query.filter_by(
                project_id=project_id, 
                username=username
            ).first()
        
            if existing_member:
                print(f"❌ User is already a member of this project")
                return jsonify({
                    'status': 'error', 
                    'error': 'User is already a member of this project'
                }), 400

            # Add member to project
            project_member = ProjectMember(
                project_id=project_id,
                username=username,
                role='member'
            )
            db.session.add(project_member)
        
            # ✅ Get project details (fixed deprecation warning)
            project = db.session.get(Project, project_id)
            if not project:
                print(f"❌ Project {project_id} not found")
                return jsonify({'status': 'error', 'error': 'Project not found'}), 404
            print(f"✅ Project found: {project.name}")
            
            # Get manager info
            manager = User.query.filter_by(username=project.created_by).first()
        
        
            # ✅ FIXED: Insert notification in database
            notification = Notification(
                recipient_username=username,
                title=f"Added to Project: {project.name}",
                message=f"You have been added to '{project.name}' by {manager.full_name if manager else project.created_by}",
                notification_type='project_invite',
                is_read=False
            )
            db.session.add(notification)
        
            # Commit all changes
            db.session.commit()
            print(f"✅ Database updated: Member and notification added")

            # ✅ Send Email Notification
            email_sent = False
            try:
                print(f"\n{'='*60}")
                print(f"📧 SENDING PROJECT INVITE EMAIL")
                print(f"{'='*60}")
                print(f"To: {user.email}")
                print(f"Recipient: {user.full_name}")
                print(f"Project: {project.name}")
                print(f"Manager: {manager.full_name if manager else project.created_by}")
                print(f"{'='*60}\n")
                
                email_send = send_project_invite_email(
                    recipient_email=user.email,
                    recipient_name=user.full_name,
                    project_name=project.name,
                    project_description=project.description or "No description provided",
                    manager_name=manager.full_name if manager else project.created_by
                )
                if email_sent:
                    print("✅ Project invite email sent successfully!")
                else:
                    print("⚠️ Email function returned False - check email logs above")
                
            except Exception as email_error:
                print(f"\n{'='*60}")
                print(f"❌ EMAIL SENDING EXCEPTION")
                print(f"{'='*60}")
                print(f"Error: {str(email_error)}")
                import traceback
                traceback.print_exc()
                print(f"{'='*60}\n")
            
            return jsonify({
                'status': 'success',
                'message': f'Member {username} added to project{"" if email_sent else " (email failed)"}'
            })
        
        except Exception as e:
            db.session.rollback()
            print(f"\n{'='*60}")
            print(f"❌ ADD MEMBER ERROR")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return jsonify({'status': 'error', 'error': str(e)}), 500


    @app.route('/api/projects/<int:project_id>/members', methods=['GET'])
    def get_project_members(project_id):
        """Get project members"""
        try:
            project_members = ProjectMember.query.filter_by(project_id=project_id).all()
        
            members_data = []
            for pm in project_members:
                user = User.query.filter_by(username=pm.username).first()
                if user:
                    member_dict = user.to_dict()
                    member_dict['project_role'] = pm.role
                    members_data.append(member_dict)
        
            return jsonify({
                "members": members_data,
                "total": len(members_data),
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/api/projects/<int:project_id>/members/<username>', methods=['DELETE'])
    def remove_project_member(project_id, username):
        """Remove member from project"""
        try:
            print(f"\n{'='*60}")
            print(f"🗑️ REMOVE MEMBER REQUEST")
            print(f"{'='*60}")
            print(f"Project ID: {project_id}")
            print(f"Username: {username}")
            
            # Get user info before removal (for email)
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify({
                    'status': 'error',
                    'error': 'User not found'
                }), 404

            # Get project info
            project = db.session.get(Project, project_id)
            if not project:
                return jsonify({
                    'status': 'error',
                    'error': 'Project not found'
                }), 404

            # Get manager info
            manager = User.query.filter_by(username=project.created_by).first()
            
            # Find the project member relationship
            project_member = ProjectMember.query.filter_by(
                project_id=project_id,
                username=username
            ).first()
        
            if not project_member:
                return jsonify({
                    'status': 'error',
                    'error': 'Member not found in this project'
                }), 404
        
            # Don't allow removing the project lead
            if project_member.role == 'lead':
                return jsonify({
                    'status': 'error',
                    'error': 'Cannot remove project lead'
                }), 403
        
            # Remove the member
            db.session.delete(project_member)
            db.session.commit()

            print(f"✅ Member removed from project")

             # Send removal email
            try:
                print(f"\n{'='*60}")
                print(f"📧 SENDING REMOVAL NOTIFICATION EMAIL")
                print(f"{'='*60}")
                print(f"To: {user.email}")
                print(f"Project: {project.name}")
                print(f"{'='*60}\n")
            
                email_sent = send_project_removal_email(
                    recipient_email=user.email,
                    recipient_name=user.full_name,
                    project_name=project.name,
                    manager_name=manager.full_name if manager else project.created_by
                )
            
                if email_sent:
                    print(f"✅ Removal notification email sent")
                else:
                    print(f"⚠️ Email sending failed")
                
            except Exception as email_error:
                print(f"❌ Email error: {str(email_error)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'status': 'success',
                    'message': f'Member {username} removed from project'
                })

            return jsonify({
                'status': 'success',
                'message': f'Member {username} removed from project'
            })

        except Exception as e:
            db.session.rollback()
            print(f"Error removing member: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500


    @app.route('/api/notifications/<username>', methods=['GET'])
    def get_notifications(username):
        """Get notifications"""
        try:
            notifications = Notification.query.filter_by(
                recipient_username=username
            ).order_by(Notification.created_at.desc()).all()
        
            unread_count = len([n for n in notifications if not n.is_read])
        
            return jsonify({
                "notifications": [n.to_dict() for n in notifications],
                "unread_count": unread_count,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500


    @app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
    def mark_notification_read(notification_id):
        """Mark notification as read"""
        try:
            notification = Notification.query.get(notification_id)
            if notification:
                notification.is_read = True
                db.session.commit()
        
            return jsonify({
                "message": "Notification marked as read",
                "status": "success"
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500 

    def send_email(to_email, subject, html_body, text_body):
        """Base function to send emails"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            print(f"📧 Attempting to send email to: {to_email}")
            print(f"📧 Subject: {subject}")

            # ✅ UPDATE THESE WITH YOUR CREDENTIALs
            smtp_server = "smtp.gmail.com"
            smtp_port = 587  # SSL port
            sender_email = "roobabaskaran194@gmail.com"  # Your Gmail
            sender_password = "pyflxnrrbpfssvwh"  # Your App Password
        
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"Agile Nexus <{sender_email}>"
            msg['To'] = to_email
        
            # Attach both text and HTML versions
            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)
        
            # Send email using SSL
            print(f"📧 Connecting to {smtp_server}:{smtp_port}...")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.ehlo()  # Identify ourselves
            print(f"📧 Starting TLS encryption...")
            server.starttls()  # Secure the connection
            server.ehlo()  # Re-identify after TLS
            print(f"📧 Logging in as {sender_email}...")
            server.login(sender_email, sender_password)
            print(f"📧 Sending message...")
            server.send_message(msg)
            server.quit()
        
            print(f"✅ Email sent successfully to {to_email}")
            return True
        
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ SMTP Authentication failed: {str(e)}")
            print("⚠️ Check your Gmail App Password!")
            return False
        except Exception as e:
            print(f"❌ Email sending failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    @app.route('/test-email')
    def test_email():
        """Test email functionality"""
        try:
            result = send_email(
                to_email="roobabaskaran194@gmail.com",  # Your email
                subject="🧪 Test Email from Agile Nexus",
                html_body="<h1>Test Email</h1><p>If you received this, email is working!</p>",
                text_body="Test Email - If you received this, email is working!"
            )
        
            return jsonify({
                "status": "success" if result else "failed",
                "message": "Check your email and Flask console for details"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500

    def send_task_assignment_email(user, task, project_name=None):
        """Send email when task is assigned"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
        
            # ✅ UPDATE THESE WITH YOUR REAL CREDENTIALS
            sender_email = "roobabaskaran194@gmail.com"
            sender_password = "pyflxnrrbpfssvwh"


            # Get project info if project_id exists
            if not project_name and task.project_id:
                project = db.session.get(Project, task.project_id)
                project_name = project.name if project else "Unknown Project"
        
            subject = f"🎯 New Task: {task.title}" + (f" - {project_name}" if project_name else "")
        
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🎯 New Task: {task.title}"
            message["From"] = sender_email
            message["To"] = user.email
        
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
                    <h2 style="color: #667eea;">🎯 New Task Assigned</h2>
                
                    <p>Hi <strong>{user.full_name}</strong>,</p>
                
                    <div style="background: #f9fafb; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3 style="margin-top: 0;">{task.title}</h3>
                        {f'<p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">📁 Project: {project_name}</p>' if project_name else ''}
                        <p style="color: #6b7280;">{task.description or 'No description'}</p>
                        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 15px 0;">
                        <div style="display: flex; gap: 20px;">
                            <div>
                                <strong>Priority:</strong> 
                                <span style="padding: 4px 8px; border-radius: 4px; background: {'#fee2e2' if task.priority == 'high' else '#fef3c7' if task.priority == 'medium' else '#dcfce7'}; color: {'#dc2626' if task.priority == 'high' else '#d97706' if task.priority == 'medium' else '#16a34a'}; font-size: 12px; font-weight: 600;">
                                    {task.priority.upper()}
                                </span>
                            </div>
                            <div>
                                <strong>Due:</strong> {task.due_date.strftime('%B %d, %Y') if task.due_date else 'Not set'}
                            </div>
                        </div>
                    </div>
            
                    <a href="http://localhost:5000" style="display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin-top: 10px;">
                        View Dashboard
                    </a>
            
                    <p style="margin-top: 30px; color: #6b7280; font-size: 14px;">
                        <strong>Assigned on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </p>
                </div>
            </body>
            </html>
            """
    
            text = f"""
            Hi {user.full_name},
    
            New Task Assigned{f' in {project_name}' if project_name else ''}
    
            Task: {task.title}
            Description: {task.description or 'No description'}
            Priority: {task.priority.upper()}
            Due: {task.due_date.strftime('%B %d, %Y') if task.due_date else 'Not set'}
    
            View Dashboard: http://localhost:5000
    
            Assigned on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            """
    
            message.attach(MIMEText(text, "plain"))
            message.attach(MIMEText(html, "html"))
    
            # ✅ FIXED: Use correct SMTP port
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(message)
            server.quit()
            
            print(f"✅ Task assignment email sent to {user.email}")
            return True
        
        except Exception as e:
            print(f"❌ Task email failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def send_task_completion_email(manager, task, submitted_by, description, file_name=None, links=None):
        """Send email when task is completed"""
        try:
            subject = f"✅ Task Completed: {task.title}"
        
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9fafb;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                        border-radius: 10px 10px 0 0;
                    }}
                    .content {{
                        background: white;
                        padding: 30px;
                        border-radius: 0 0 10px 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .task-card {{
                        background: #f0fdf4;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                        border-left: 4px solid #10b981;
                    }}
                    .button {{
                        display: inline-block;
                        padding: 12px 30px;
                        background: #10b981;
                        color: white;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: 600;
                        margin-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>✅ Task Completed</h1>
                    </div>
                    <div class="content">
                        <p>Hi <strong>{manager.full_name}</strong>,</p>
                    
                        <p><strong>{submitted_by}</strong> has completed the following task:</p>
                    
                        <div class="task-card">
                            <h3 style="margin-top: 0; color: #059669;">{task.title}</h3>
                            <p style="color: #6b7280;"><strong>Original Description:</strong><br>{task.description or 'No description'}</p>
                            <hr style="border: none; border-top: 1px solid #d1fae5; margin: 15px 0;">
                            <p style="color: #6b7280;"><strong>Submission Notes:</strong></p>
                            <p style="color: #374151;">{description}</p>
                            {f'<p style="color: #6b7280; margin-top: 10px;"><strong>📎 File Attached:</strong> {file_name}</p>' if file_name else ''}
                            {f'<p style="color: #6b7280; margin-top: 10px;"><strong>🔗 Links:</strong><br>{links}</p>' if links else ''}
                        </div>
                    
                        <div style="text-align: center;">
                            <a href="http://localhost:5000/manager-dashboard" class="button">
                                Review Submission
                            </a>
                        </div>

                        <p style="margin-top: 30px; color: #6b7280; font-size: 14px;">
                            <strong>Submitted by:</strong> {submitted_by}<br>
                            <strong>Submitted on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
        
            text_body = f"""
            Hi {manager.full_name},
        
            {submitted_by} has completed the task: {task.title}
        
            Submission Notes: {description}
            {'File attached: ' + file_name if file_name else ''}
            {'Links: ' + links if links else ''}
        
            Review the submission at: http://localhost:5000/manager-dashboard
        
            Submitted on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            """
        
            return send_email(manager.email, subject, html_body, text_body)
        
        except Exception as e:
            print(f"❌ Task completion email failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def send_project_invite_email(recipient_email, recipient_name, project_name, project_description, manager_name):
        """Send email when user is added to a project"""
    
        subject = f"🎉 You've been added to project: {project_name}"
    
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9fafb;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .project-card {{
                    background: #f3f4f6;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 4px solid #667eea;
                }}
                .project-name {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 10px;
                }}
                .button {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: #667eea;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    margin-top: 20px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #6b7280;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Welcome to the Team!</h1>
                </div>
                <div class="content">
                    <p>Hi <strong>{recipient_name}</strong>,</p>
                
                    <p>Great news! <strong>{manager_name}</strong> has added you to a new project on <strong>Agile Nexus</strong>.</p>
                
                    <div class="project-card">
                        <div class="project-name">📁 {project_name}</div>
                        <p style="color: #6b7280; margin: 0;">{project_description}</p>
                    </div>
                
                    <p><strong>What's Next?</strong></p>
                    <ul>
                        <li>Log in to your Agile Nexus dashboard</li>
                        <li>View project details and team members</li>
                        <li>Check assigned tasks and dependencies</li>
                        <li>Collaborate with your team</li>
                    </ul>
                
                    <div style="text-align: center;">
                        <a href="http://localhost:5000" class="button">
                            Open Dashboard
                        </a>
                    </div>
                
                    <p style="margin-top: 30px; color: #6b7280; font-size: 14px;">
                        <strong>Project Manager:</strong> {manager_name}<br>
                        <strong>Added on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from Agile Nexus</p>
                    <p>AI-Powered Sprint Health Prediction System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
        text_body = f"""
        Hi {recipient_name},
    
        You've been added to project: {project_name}
    
        Project Description: {project_description}
        Added by: {manager_name}
        Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
    
        Log in to Agile Nexus to get started: http://localhost:5000
    
        ---
        Agile Nexus - AI-Powered Sprint Health Prediction
        """
    
        send_email(recipient_email, subject, html_body, text_body)
    
    @app.route('/check-user/<username>')
    def check_user(username):
        """Check if user exists and show their details"""
        user = User.query.filter_by(username=username).first()
        if user:
            return jsonify({
                'status': 'success',
                'user': user.to_dict()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404

    def send_project_removal_email(recipient_email, recipient_name, project_name, manager_name):
        """Send email when user is removed from a project"""
    
        subject = f"🔔 Removed from project: {project_name}"
    
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9fafb;
                }}
                .header {{
                    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .project-card {{
                    background: #fef2f2;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 4px solid #ef4444;
                }}
                .project-name {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #dc2626;
                    margin-bottom: 10px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #6b7280;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔔 Project Access Removed</h1>
                </div>
                <div class="content">
                    <p>Hi <strong>{recipient_name}</strong>,</p>
                
                    <p>You have been removed from the following project by <strong>{manager_name}</strong>.</p>
                
                    <div class="project-card">
                        <div class="project-name">📁 {project_name}</div>
                    </div>
                
                    <p><strong>What This Means:</strong></p>
                    <ul>
                        <li>You no longer have access to this project</li>
                        <li>You won't receive updates about this project</li>
                        <li>Your account remains active for other projects</li>
                    </ul>
                
                    <p style="margin-top: 30px; color: #6b7280; font-size: 14px;">
                        <strong>Removed by:</strong> {manager_name}<br>
                        <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </p>
                </div>
                <div class="footer">
                    <p>This is an automated notification from Agile Nexus</p>
                    <p>AI-Powered Sprint Health Prediction System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
        text_body = f"""
        Hi {recipient_name},
    
        You have been removed from project: {project_name}
    
        Removed by: {manager_name}
        Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
    
        You no longer have access to this project, but your account remains active for other projects.
    
        ---
        Agile Nexus - AI-Powered Sprint Health Prediction
        """
    
        return send_email(recipient_email, subject, html_body, text_body)
    
    #===== INTEGRATED ROUTES ====================
    
    @app.route('/train-all-models', methods=['GET', 'POST'])
    def train_all_models():
        try:
            results = {}
            
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
            
            # Train Sprint Health Predictor
            sprint_features = processing_results.get('sample_sprint_features', [])
            sprint_predictor = SprintHealthPredictor()
            results['sprint_health'] = sprint_predictor.train(sprint_features)
            sprint_predictor.save_model()
            
            # Train Productivity Heatmap AI
            productivity_features = processing_results.get('sample_productivity_features', [])
            heatmap_ai = ProductivityHeatmapAI()
            results['productivity_heatmap'] = heatmap_ai.train(productivity_features)
            heatmap_ai.save_model()
            
            # Train Dependency Tracker
            dependency_features = processing_results.get('sample_dependency_features', [])
            dependency_tracker = DependencyTracker()
            results['dependency_tracker'] = dependency_tracker.train(dependency_features)
            dependency_tracker.save_model()
            
            return jsonify({
                "message": "All AI models trained successfully!",
                "training_results": results,
                "models_trained": [
                    "Sprint Health Predictor (Random Forest + XGBoost)",
                    "Productivity Heatmap AI (K-Means + SVR)",
                    "Dependency Tracker (Graph Neural Networks)"
                ],
                "status": "success"
            })
            
        except Exception as e:
            return jsonify({
                "message": "Model training failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    # ==================== FULL PIPELINE ====================
    
    @app.route('/run-full-pipeline', methods=['GET', 'POST'])
    def run_full_pipeline():
        try:
            results = {}
            
            # Step 1: ETL
            etl = ETLCoordinator(app.config)
            etl_results = etl.run_full_extraction(days_back=7)
            results['etl'] = etl_results
            
            # Step 2: Processing
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
            results['processing'] = processing_results
            
            # Step 3: Train all models
            sprint_features = processing_results.get('sample_sprint_features', [])
            productivity_features = processing_results.get('sample_productivity_features', [])
            dependency_features = processing_results.get('sample_dependency_features', [])
            
            sprint_predictor = SprintHealthPredictor()
            heatmap_ai = ProductivityHeatmapAI()
            dependency_tracker = DependencyTracker()
            
            sprint_training = sprint_predictor.train(sprint_features)
            productivity_training = heatmap_ai.train(productivity_features)
            dependency_training = dependency_tracker.train(dependency_features)
            
            results['model_training'] = {
                'sprint_health': sprint_training,
                'productivity_heatmap': productivity_training,
                'dependency_tracker': dependency_training
            }
            
            # Step 4: Generate predictions
            sprint_predictions = sprint_predictor.predict(sprint_features[:2] if sprint_features else [])
            productivity_predictions = heatmap_ai.predict(productivity_features[:2] if productivity_features else [])
            dependency_predictions = dependency_tracker.predict(dependency_features[:2] if dependency_features else [])
            
            results['predictions'] = {
                'sprint_health': sprint_predictions,
                'productivity': productivity_predictions,
                'dependencies': dependency_predictions
            }
            
            # Step 5: Generate visualizations
            team_heatmap = heatmap_ai.get_team_heatmap()
            dependency_network = dependency_tracker.get_dependency_visualization()
            
            results['visualizations'] = {
                'team_heatmap': team_heatmap,
                'dependency_network': dependency_network
            }
            
            return jsonify({
                "message": "Full Agile Nexus pipeline completed successfully!",
                "pipeline_results": results,
                "components_executed": [
                    "ETL Pipeline",
                    "Feature Engineering",
                    "Sprint Health Predictor (Random Forest + XGBoost)",
                    "Productivity Heatmap AI (K-Means + SVR)",
                    "Dependency Tracker (Graph Neural Networks)"
                ],
                "status": "success"
            })
            
        except Exception as e:
            return jsonify({
                "message": "Full pipeline failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    # ==================== DEBUG ROUTES ====================
    
    @app.route('/debug-processing')
    def debug_processing():
        try:
            processor = ProcessingCoordinator()
            results = processor.run_full_processing(days_back=7)
            
            return jsonify({
                "message": "Debug processing data",
                "results_keys": list(results.keys()) if isinstance(results, dict) else "Not a dict",
                "results_sample": str(results)[:500] + "..." if len(str(results)) > 500 else str(results),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "Debug failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    @app.route('/api-status')
    def api_status():
        return jsonify({
            "agile_nexus_status": "operational",
            "ai_models": {
                "sprint_health_predictor": "Random Forest + XGBoost",
                "productivity_heatmap_ai": "K-Means + SVR",
                "dependency_tracker": "Graph Neural Networks"
            },
            "endpoints_available": 18,
            "database_connected": True,
            "version": "2.0.0",
            "status": "success"
        })
    # ==================== ENHANCED API ENDPOINTS ====================

    @app.route('/sprint-risk-summary')
    def sprint_risk_summary():
        try:
            processor = ProcessingCoordinator()
            processing_results = processor.run_full_processing(days_back=7)
        
            sprint_features = processing_results.get('sample_sprint_features', [])
        
            if not sprint_features:
                return jsonify({
                    "message": "No sprint data available",
                    "status": "error"
                }), 404
        
            predictor = SprintHealthPredictor()
            predictions = predictor.predict(sprint_features)
        
            # Calculate summary statistics
            if 'predictions' in predictions:
                risks = [p['predicted_risk_score'] for p in predictions['predictions']]
                summary = {
                    'total_users': len(risks),
                    'average_risk': round(sum(risks) / len(risks), 1) if risks else 0,
                    'high_risk_users': len([r for r in risks if r > 70]),
                    'medium_risk_users': len([r for r in risks if 50 < r <= 70]),
                    'low_risk_users': len([r for r in risks if r <= 50]),
                    'users_at_risk': predictions['predictions']
                }
            else:
                summary = {'error': 'No predictions available'}
        
            return jsonify({
                "message": "Sprint risk summary generated",
                "summary": summary,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "message": "Risk summary failed",
                "error": str(e),
                "status": "error"
            }), 500  

    @app.route('/productivity-recommendations')
    def productivity_recommendations():
        try:
            heatmap_ai = ProductivityHeatmapAI()
        
            # Generate sample data for current week
            current_hour = 14  # 2 PM
            sample_data = [{
                'user_id': 'current_user',
                'hour_of_day': current_hour,
                'day_of_week': 2,  # Wednesday
                'meeting_count': 2,
                'focus_events': 1,
                'calendar_density': 0.7,
                'message_activity': 30,
                'commit_activity': 2
            }]
        
            predictions = heatmap_ai.predict(sample_data)
        
            recommendations = {
                'current_time_analysis': predictions.get('predictions', []),
                'best_times_today': [],
                'scheduling_tips': [
                    "Schedule important tasks during peak productivity hours (9-11 AM)",
                    "Keep afternoons lighter for collaborative work",
                    "Block focus time when productivity is highest"
                ]
            }
        
            return jsonify({
                "message": "Productivity recommendations generated",
                "recommendations": recommendations,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "message": "Recommendations failed",
                "error": str(e),
                "status": "error"
            }), 500 
    
    @app.route('/dependency-risk-report')
    def dependency_risk_report():
        try:
            # Sample critical dependencies
            critical_deps = [
                {
                    'source_story': 'CRITICAL_FEATURE_A',
                    'dependent_story': 'RELEASE_BLOCKER_B',
                    'dependency_type': 'blocks',
                    'risk_probability': 0.85,
                    'cascade_impact': 8
                },
                {
                    'source_story': 'RELEASE_BLOCKER_B',
                    'dependent_story': 'CUSTOMER_FEATURE_C',
                    'dependency_type': 'blocks',
                    'risk_probability': 0.6,
                    'cascade_impact': 5
                }
            ]
        
            dependency_tracker = DependencyTracker()
            predictions = dependency_tracker.predict(critical_deps)
        
            # Generate report
            report = {
                'total_dependencies': len(critical_deps),
                'high_risk_count': len([d for d in critical_deps if d['risk_probability'] > 0.7]),
                'critical_path_items': predictions.get('predictions', []),
                'action_items': [
                    "Address high-risk dependencies immediately",
                    "Consider parallel development for blocked items",
                    "Schedule dependency review meeting"
                ]
            }
        
            return jsonify({
                "message": "Dependency risk report generated",
                "report": report,
                "status": "success"
            })
        
        except Exception as e:
            return jsonify({
                "message": "Report generation failed",
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/health-check')
    def health_check():
        try:
            # Check database
            db.session.execute(db.text('SELECT 1'))
            db_status = "healthy"
        except:
            db_status = "unhealthy"
    
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "api_version": "2.0.0",
            "models_available": [
                "sprint_health_predictor",
                "productivity_heatmap_ai",
                "dependency_tracker"
            ]
        })
    @app.route('/dashboard')
    def serve_dashboard():
        return send_file('../frontend/dashboard/index.html')
    @app.route('/run-real-etl', methods=['GET', 'POST'])  # ✅ Allow both methods
    def run_real_etl():
        """Extract from real APIs (Google Calendar, Slack, GitHub)"""
        try:
            from etl_pipeline.real_api_coordinator import RealAPICoordinator
            from models import CalendarData, CodeActivity, CommunicationData
        
            coordinator = RealAPICoordinator()
        
            github_users = ['Rooba8925', 'praneetaAD078', 'sadhana-095']
        
            # Run extraction
            results = coordinator.run_real_extraction(
                days_back=365,
                github_usernames=github_users
            )
        
            # Get actual database counts
            real_calendar = CalendarData.query.filter_by(data_source='real').count()
            mock_calendar = CalendarData.query.filter_by(data_source='mock').count()
            total_calendar = CalendarData.query.count()
        
            real_comm = CommunicationData.query.filter_by(data_source='real').count()
            mock_comm = CommunicationData.query.filter_by(data_source='mock').count()
            total_comm = CommunicationData.query.count()
        
            real_code = CodeActivity.query.filter_by(data_source='real').count()
            mock_code = CodeActivity.query.filter_by(data_source='mock').count()
            total_code = CodeActivity.query.count()
        
            # Calculate percentages
            cal_percentage = round((real_calendar / total_calendar * 100) if total_calendar > 0 else 0, 1)
            comm_percentage = round((real_comm / total_comm * 100) if total_comm > 0 else 0, 1)
            code_percentage = round((real_code / total_code * 100) if total_code > 0 else 0, 1)
        
            return jsonify({
                "status": "success",
                "message": "Real API extraction completed!",
                
                "calendar_events": real_calendar,
                "communication_records": real_comm,
                "code_activities": real_code,
                
                "database_summary": {
                    "calendar": {
                        "real": real_calendar,
                        "mock": mock_calendar,
                        "total": total_calendar,
                        "real_percentage": f"{cal_percentage}%"
                    },
                    "communication": {
                        "real": real_comm,
                        "mock": mock_comm,
                        "total": total_comm,
                        "real_percentage": f"{comm_percentage}%"
                    },
                    "code_activity": {
                        "real": real_code,
                        "mock": mock_code,
                        "total": total_code,
                        "real_percentage": f"{code_percentage}%"
                    }
                },
                
                "errors": results.get('errors', [])
            })
        
        except Exception as e:
            import traceback
            return jsonify({
                "status": "error",
                "message": "Real API extraction failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    @app.route('/test-real-apis')
    def test_real_apis():
        """Check which real APIs are configured"""
        try:
            coordinator = RealAPICoordinator()
        
            status = coordinator.get_api_status()

        
            return jsonify({
                "message": "Real API configuration status",
                "configured_apis": status,
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "message": "API status check failed",
                "error": str(e),
                "status": "error"
            }), 500
    
    return app
app = create_app()
if __name__ == '__main__':
    
    
    # Create all tables
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    # Run the app
    app.run(debug=True, host='localhost', port=5000)