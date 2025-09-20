/**
 * Y.M.I.R Firebase Authentication System
 * Handles user authentication, temporary users, and data persistence
 */

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyBJ9FyFsGvyn1eWbj2tM_xBpUmSwf-miLg",
    authDomain: "y-m-i-r.firebaseapp.com",
    projectId: "y-m-i-r",
    storageBucket: "y-m-i-r.firebasestorage.app",
    messagingSenderId: "40657219852",
    appId: "1:40657219852:web:52d084407c9cf7aee49929",
    measurementId: "G-BCD3S2HTDJ"
};

// Firebase imports (will be loaded dynamically)
let firebase, auth, db, analytics;

// User state management
class YMIRUserManager {
    constructor() {
        this.currentUser = null;
        this.isTemporaryUser = true;
        this.temporaryData = {
            emotions: [],
            conversations: [],
            musicHistory: [],
            preferences: {},
            sessionId: this.generateSessionId()
        };
        this.initializeFirebase();
    }

    // Generate unique session ID for temporary users
    generateSessionId() {
        return 'temp_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Initialize Firebase
    async initializeFirebase() {
        try {
            console.log('🔥 Initializing Firebase...');
            
            // Dynamic import of Firebase modules
            const { initializeApp } = await import('https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js');
            const { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, onAuthStateChanged, GoogleAuthProvider, signInWithPopup } = await import('https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js');
            const { getFirestore, doc, setDoc, getDoc, updateDoc, collection, addDoc, query, where, orderBy, getDocs } = await import('https://www.gstatic.com/firebasejs/10.8.0/firebase-firestore.js');
            const { getAnalytics } = await import('https://www.gstatic.com/firebasejs/10.8.0/firebase-analytics.js');

            // Initialize Firebase app
            const app = initializeApp(firebaseConfig);
            
            // Initialize services
            this.auth = getAuth(app);
            this.db = getFirestore(app);
            this.analytics = getAnalytics(app);
            
            // Store Firebase functions for later use
            this.firebaseFunctions = {
                signInWithEmailAndPassword,
                createUserWithEmailAndPassword,
                signOut,
                onAuthStateChanged,
                GoogleAuthProvider,
                signInWithPopup,
                doc,
                setDoc,
                getDoc,
                updateDoc,
                collection,
                addDoc,
                query,
                where,
                orderBy,
                getDocs
            };

            console.log('✅ Firebase initialized successfully');
            
            // Set up auth state listener
            this.setupAuthStateListener();
            
            // Initialize UI
            this.initializeAuthUI();
            
        } catch (error) {
            console.error('❌ Firebase initialization failed:', error);
            this.showError('Firebase initialization failed. Some features may be limited.');
        }
    }

    // Set up authentication state listener
    setupAuthStateListener() {
        this.firebaseFunctions.onAuthStateChanged(this.auth, (user) => {
            if (user) {
                console.log('👤 User signed in:', user.email);
                this.currentUser = user;
                this.isTemporaryUser = false;
                this.onUserSignIn(user);
            } else {
                console.log('👤 User signed out, switching to temporary mode');
                this.currentUser = null;
                this.isTemporaryUser = true;
                this.onUserSignOut();
            }
        });
    }

    // Initialize authentication UI
    initializeAuthUI() {
        this.createAuthModal();
        this.createUserStatusUI();
        this.updateAuthUI();
    }

    // Create authentication modal
    createAuthModal() {
        const modalHTML = `
            <div id="authModal" class="auth-modal hidden">
                <div class="auth-modal-content">
                    <div class="auth-modal-header">
                        <h2>🚀 Join Y.M.I.R</h2>
                        <button id="closeAuthModal" class="close-button">&times;</button>
                    </div>
                    
                    <div class="auth-tabs">
                        <button id="signInTab" class="auth-tab active">Sign In</button>
                        <button id="signUpTab" class="auth-tab">Sign Up</button>
                    </div>
                    
                    <div id="signInForm" class="auth-form">
                        <h3>Welcome Back!</h3>
                        <p class="auth-description">Sign in to save your emotional journey and personalized recommendations.</p>
                        
                        <div class="form-group">
                            <label for="signInEmail">Email</label>
                            <input type="email" id="signInEmail" placeholder="your.email@example.com" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="signInPassword">Password</label>
                            <input type="password" id="signInPassword" placeholder="Your password" required>
                        </div>
                        
                        <button id="signInButton" class="auth-button primary">Sign In</button>
                        <button id="googleSignInButton" class="auth-button google">
                            <span class="google-icon">🔍</span> Continue with Google
                        </button>
                        
                        <div class="auth-footer">
                            <a href="#" id="forgotPassword">Forgot password?</a>
                        </div>
                    </div>
                    
                    <div id="signUpForm" class="auth-form hidden">
                        <h3>Join Y.M.I.R</h3>
                        <p class="auth-description">Create an account to unlock personalized mental health support and save your progress.</p>
                        
                        <div class="form-group">
                            <label for="signUpName">Full Name</label>
                            <input type="text" id="signUpName" placeholder="Your name" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="signUpEmail">Email</label>
                            <input type="email" id="signUpEmail" placeholder="your.email@example.com" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="signUpPassword">Password</label>
                            <input type="password" id="signUpPassword" placeholder="Create a strong password" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="confirmPassword">Confirm Password</label>
                            <input type="password" id="confirmPassword" placeholder="Confirm your password" required>
                        </div>
                        
                        <div class="form-group checkbox">
                            <input type="checkbox" id="agreeTerms" required>
                            <label for="agreeTerms">I agree to the <a href="#" target="_blank">Terms of Service</a> and <a href="#" target="_blank">Privacy Policy</a></label>
                        </div>
                        
                        <button id="signUpButton" class="auth-button primary">Create Account</button>
                        <button id="googleSignUpButton" class="auth-button google">
                            <span class="google-icon">🔍</span> Continue with Google
                        </button>
                    </div>
                    
                    <div id="authLoading" class="auth-loading hidden">
                        <div class="loading-spinner"></div>
                        <p>Authenticating...</p>
                    </div>
                    
                    <div id="authError" class="auth-error hidden"></div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.setupAuthModalEvents();
    }

    // Create user status UI in header - DISABLED: Using header navigation auth instead
    createUserStatusUI() {
        // Skip creating duplicate user status UI since AI dashboard uses header navigation
        console.log('🔗 Skipping duplicate user status UI - using header navigation authentication');
        return;
    }

    // Setup authentication modal events
    setupAuthModalEvents() {
        // Tab switching
        document.getElementById('signInTab').addEventListener('click', () => this.switchAuthTab('signIn'));
        document.getElementById('signUpTab').addEventListener('click', () => this.switchAuthTab('signUp'));
        
        // Close modal
        document.getElementById('closeAuthModal').addEventListener('click', () => this.closeAuthModal());
        document.getElementById('authModal').addEventListener('click', (e) => {
            if (e.target.id === 'authModal') this.closeAuthModal();
        });
        
        // Sign in form
        document.getElementById('signInButton').addEventListener('click', () => this.handleSignIn());
        document.getElementById('signInForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleSignIn();
        });
        
        // Sign up form
        document.getElementById('signUpButton').addEventListener('click', () => this.handleSignUp());
        document.getElementById('signUpForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleSignUp();
        });
        
        // Google authentication
        document.getElementById('googleSignInButton').addEventListener('click', () => this.handleGoogleSignIn());
        document.getElementById('googleSignUpButton').addEventListener('click', () => this.handleGoogleSignIn());
        
        // Enter key handling
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !document.getElementById('authModal').classList.contains('hidden')) {
                const activeTab = document.querySelector('.auth-tab.active').id;
                if (activeTab === 'signInTab') {
                    this.handleSignIn();
                } else {
                    this.handleSignUp();
                }
            }
        });
    }

    // Setup user status events - DISABLED: Using header navigation auth instead
    setupUserStatusEvents() {
        // Skip setting up events for UI elements we're not creating
        console.log('🔗 Skipping user status events - using header navigation authentication');
        return;
    }

    // Switch authentication tabs
    switchAuthTab(tab) {
        const signInTab = document.getElementById('signInTab');
        const signUpTab = document.getElementById('signUpTab');
        const signInForm = document.getElementById('signInForm');
        const signUpForm = document.getElementById('signUpForm');
        
        if (tab === 'signIn') {
            signInTab.classList.add('active');
            signUpTab.classList.remove('active');
            signInForm.classList.remove('hidden');
            signUpForm.classList.add('hidden');
        } else {
            signUpTab.classList.add('active');
            signInTab.classList.remove('active');
            signUpForm.classList.remove('hidden');
            signInForm.classList.add('hidden');
        }
    }

    // Open authentication modal
    openAuthModal() {
        document.getElementById('authModal').classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    // Close authentication modal
    closeAuthModal() {
        document.getElementById('authModal').classList.add('hidden');
        document.body.style.overflow = 'auto';
        this.clearAuthErrors();
    }

    // Handle email/password sign in
    async handleSignIn() {
        const email = document.getElementById('signInEmail').value.trim();
        const password = document.getElementById('signInPassword').value;
        
        if (!email || !password) {
            this.showAuthError('Please enter both email and password.');
            return;
        }
        
        try {
            this.showAuthLoading(true);
            await this.firebaseFunctions.signInWithEmailAndPassword(this.auth, email, password);
            this.closeAuthModal();
            this.showSuccess('Welcome back! Your data has been restored.');
        } catch (error) {
            this.showAuthError(this.getAuthErrorMessage(error));
        } finally {
            this.showAuthLoading(false);
        }
    }

    // Handle email/password sign up
    async handleSignUp() {
        const name = document.getElementById('signUpName').value.trim();
        const email = document.getElementById('signUpEmail').value.trim();
        const password = document.getElementById('signUpPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const agreeTerms = document.getElementById('agreeTerms').checked;
        
        // Validation
        if (!name || !email || !password || !confirmPassword) {
            this.showAuthError('Please fill in all required fields.');
            return;
        }
        
        if (password !== confirmPassword) {
            this.showAuthError('Passwords do not match.');
            return;
        }
        
        if (password.length < 6) {
            this.showAuthError('Password must be at least 6 characters long.');
            return;
        }
        
        if (!agreeTerms) {
            this.showAuthError('Please agree to the Terms of Service and Privacy Policy.');
            return;
        }
        
        try {
            this.showAuthLoading(true);
            const userCredential = await this.firebaseFunctions.createUserWithEmailAndPassword(this.auth, email, password);
            
            // Create user profile
            await this.createUserProfile(userCredential.user, { name, email });
            
            this.closeAuthModal();
            this.showSuccess('Account created successfully! Welcome to Y.M.I.R.');
        } catch (error) {
            this.showAuthError(this.getAuthErrorMessage(error));
        } finally {
            this.showAuthLoading(false);
        }
    }

    // Handle Google sign in
    async handleGoogleSignIn() {
        try {
            this.showAuthLoading(true);
            const provider = new this.firebaseFunctions.GoogleAuthProvider();
            const result = await this.firebaseFunctions.signInWithPopup(this.auth, provider);
            
            // Create or update user profile
            await this.createUserProfile(result.user, {
                name: result.user.displayName,
                email: result.user.email
            });
            
            this.closeAuthModal();
            this.showSuccess('Successfully signed in with Google!');
        } catch (error) {
            this.showAuthError(this.getAuthErrorMessage(error));
        } finally {
            this.showAuthLoading(false);
        }
    }

    // Handle sign out
    async handleSignOut() {
        try {
            await this.firebaseFunctions.signOut(this.auth);
            this.showSuccess('You have been signed out. Your session data is preserved temporarily.');
        } catch (error) {
            this.showError('Error signing out: ' + error.message);
        }
    }

    // Create user profile in Firestore
    async createUserProfile(user, userData) {
        try {
            const userRef = this.firebaseFunctions.doc(this.db, 'users', user.uid);
            const userDoc = await this.firebaseFunctions.getDoc(userRef);
            
            if (!userDoc.exists()) {
                // New user - create profile and migrate temporary data
                await this.firebaseFunctions.setDoc(userRef, {
                    name: userData.name,
                    email: userData.email,
                    createdAt: new Date(),
                    preferences: this.temporaryData.preferences,
                    lastActive: new Date()
                });
                
                // Migrate temporary data if exists
                if (this.temporaryData.emotions.length > 0 || this.temporaryData.conversations.length > 0) {
                    await this.migrateTemporaryData(user.uid);
                }
            } else {
                // Existing user - update last active
                await this.firebaseFunctions.updateDoc(userRef, {
                    lastActive: new Date()
                });
            }
        } catch (error) {
            console.error('Error creating user profile:', error);
        }
    }

    // Migrate temporary data to authenticated user
    async migrateTemporaryData(userId) {
        try {
            console.log('🔄 Migrating temporary data to authenticated user...');
            
            // Migrate emotions data
            if (this.temporaryData.emotions.length > 0) {
                const emotionsRef = this.firebaseFunctions.collection(this.db, 'users', userId, 'emotions');
                for (const emotion of this.temporaryData.emotions) {
                    await this.firebaseFunctions.addDoc(emotionsRef, emotion);
                }
            }
            
            // Migrate conversations
            if (this.temporaryData.conversations.length > 0) {
                const conversationsRef = this.firebaseFunctions.collection(this.db, 'users', userId, 'conversations');
                for (const conversation of this.temporaryData.conversations) {
                    await this.firebaseFunctions.addDoc(conversationsRef, conversation);
                }
            }
            
            // Migrate music history
            if (this.temporaryData.musicHistory.length > 0) {
                const musicRef = this.firebaseFunctions.collection(this.db, 'users', userId, 'musicHistory');
                for (const music of this.temporaryData.musicHistory) {
                    await this.firebaseFunctions.addDoc(musicRef, music);
                }
            }
            
            console.log('✅ Temporary data migrated successfully');
            this.clearTemporaryData();
        } catch (error) {
            console.error('❌ Error migrating temporary data:', error);
        }
    }

    // Clear temporary data after migration
    clearTemporaryData() {
        this.temporaryData = {
            emotions: [],
            conversations: [],
            musicHistory: [],
            preferences: {},
            sessionId: this.generateSessionId()
        };
    }

    // Handle user sign in event
    async onUserSignIn(user) {
        this.updateAuthUI();
        await this.loadUserData(user.uid);
        this.triggerUserStateChange('authenticated');
    }

    // Handle user sign out event
    onUserSignOut() {
        this.updateAuthUI();
        this.triggerUserStateChange('temporary');
    }

    // Update authentication UI
    updateAuthUI() {
        const tempBanner = document.getElementById('temporaryUserBanner');
        const authArea = document.getElementById('authenticatedUserArea');
        const userName = document.getElementById('userName');
        
        if (this.isTemporaryUser) {
            tempBanner?.classList.remove('hidden');
            authArea?.classList.add('hidden');
        } else {
            tempBanner?.classList.add('hidden');
            authArea?.classList.remove('hidden');
            if (userName && this.currentUser) {
                userName.textContent = this.currentUser.displayName || this.currentUser.email.split('@')[0];
            }
        }
    }

    // Load user data from Firestore
    async loadUserData(userId) {
        try {
            console.log('📊 Loading user data...');
            
            // Load user preferences
            const userRef = this.firebaseFunctions.doc(this.db, 'users', userId);
            const userDoc = await this.firebaseFunctions.getDoc(userRef);
            
            if (userDoc.exists()) {
                const userData = userDoc.data();
                this.temporaryData.preferences = userData.preferences || {};
            }
            
            console.log('✅ User data loaded successfully');
        } catch (error) {
            console.error('❌ Error loading user data:', error);
        }
    }

    // Save data (emotion, conversation, music history)
    async saveData(type, data) {
        if (this.isTemporaryUser) {
            // Save to temporary storage
            this.temporaryData[type].push({
                ...data,
                timestamp: new Date(),
                sessionId: this.temporaryData.sessionId
            });
            console.log(`📝 Saved ${type} data to temporary storage`);
        } else {
            // Save to Firestore
            try {
                const collectionRef = this.firebaseFunctions.collection(this.db, 'users', this.currentUser.uid, type);
                await this.firebaseFunctions.addDoc(collectionRef, {
                    ...data,
                    timestamp: new Date()
                });
                console.log(`✅ Saved ${type} data to Firestore`);
            } catch (error) {
                console.error(`❌ Error saving ${type} data:`, error);
                // Fallback to temporary storage
                this.temporaryData[type].push({
                    ...data,
                    timestamp: new Date(),
                    sessionId: this.temporaryData.sessionId
                });
            }
        }
    }

    // Get user data
    getUserData(type = null) {
        if (type) {
            return this.temporaryData[type] || [];
        }
        return this.temporaryData;
    }

    // Get current user status
    getUserStatus() {
        return {
            isAuthenticated: !this.isTemporaryUser,
            user: this.currentUser,
            sessionId: this.temporaryData.sessionId,
            hasTemporaryData: this.temporaryData.emotions.length > 0 || 
                             this.temporaryData.conversations.length > 0 || 
                             this.temporaryData.musicHistory.length > 0
        };
    }

    // Trigger user state change event
    triggerUserStateChange(state) {
        const event = new CustomEvent('userStateChange', {
            detail: {
                state: state,
                user: this.currentUser,
                isTemporary: this.isTemporaryUser,
                data: this.temporaryData
            }
        });
        document.dispatchEvent(event);
    }

    // Utility functions
    showAuthLoading(show) {
        const loading = document.getElementById('authLoading');
        const forms = document.querySelectorAll('.auth-form');
        
        if (show) {
            loading?.classList.remove('hidden');
            forms.forEach(form => form.classList.add('hidden'));
        } else {
            loading?.classList.add('hidden');
            const activeTab = document.querySelector('.auth-tab.active').id;
            const activeForm = activeTab === 'signInTab' ? 'signInForm' : 'signUpForm';
            document.getElementById(activeForm)?.classList.remove('hidden');
        }
    }

    showAuthError(message) {
        const errorDiv = document.getElementById('authError');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.classList.remove('hidden');
        }
    }

    clearAuthErrors() {
        const errorDiv = document.getElementById('authError');
        if (errorDiv) {
            errorDiv.classList.add('hidden');
            errorDiv.textContent = '';
        }
    }

    showSuccess(message) {
        // Create or update success notification
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 4000);
    }

    getAuthErrorMessage(error) {
        switch (error.code) {
            case 'auth/user-not-found':
                return 'No account found with this email address.';
            case 'auth/wrong-password':
                return 'Incorrect password. Please try again.';
            case 'auth/email-already-in-use':
                return 'An account with this email already exists.';
            case 'auth/weak-password':
                return 'Password is too weak. Please choose a stronger password.';
            case 'auth/invalid-email':
                return 'Please enter a valid email address.';
            case 'auth/popup-closed-by-user':
                return 'Sign-in was cancelled.';
            default:
                return error.message || 'An error occurred. Please try again.';
        }
    }
}

// Initialize user manager when DOM is loaded
let ymirUserManager;

document.addEventListener('DOMContentLoaded', () => {
    ymirUserManager = new YMIRUserManager();
    
    // Make it globally accessible
    window.ymirAuth = ymirUserManager;
    
    console.log('🚀 Y.M.I.R Authentication System Initialized');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = YMIRUserManager;
}