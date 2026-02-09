/**
 * Voice Recognition Client
 * A JavaScript client for voice recognition services
 */
class VoiceRecognitionClient {
    /**
     * Create a new VoiceRecognitionClient
     * @param {string} apiBaseUrl - Base URL for the voice recognition API
     */
    constructor(apiBaseUrl = '/api/voice') {
        this.apiBaseUrl = apiBaseUrl;
        this.isListening = false;
        this.selectedMicrophoneIndex = null;
        this.onRecognitionResult = null;
        this.onError = null;
        this.onStatusChange = null;
        this.currentStream = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        console.log('Voice Recognition Client initialized with API base URL:', apiBaseUrl);
    }

    /**
     * Set callback for recognition results
     * @param {Function} callback - Function to call with recognition results
     */
    setRecognitionResultCallback(callback) {
        this.onRecognitionResult = callback;
    }
    
    /**
     * Set callback for error handling
     * @param {Function} callback - Function to call with error messages
     */
    setErrorCallback(callback) {
        this.onError = callback;
    }
    
    /**
     * Set callback for status changes
     * @param {Function} callback - Function to call with status updates
     */
    setStatusCallback(callback) {
        this.onStatusChange = callback;
    }

    /**
     * Report status changes internally and to callback if set
     * @param {string} status - Status message
     * @private
     */
    _reportStatus(status) {
        console.log(`Voice client status: ${status}`);
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }

    /**
     * Report errors internally and to callback if set
     * @param {string} error - Error message
     * @private
     */
    _reportError(error) {
        console.error(`Voice client error: ${error}`);
        if (this.onError) {
            this.onError(error);
        }
    }
    
    /**
     * Get available microphones
     * @returns {Promise<Array>} - Array of available microphone devices
     */
    async getMicrophones() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                throw new Error('Media devices API not supported in this browser');
            }
            
            // Check if we need permission first
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop());
            } catch (err) {
                throw new Error('Microphone permission denied');
            }
            
            // Get devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            const microphones = devices.filter(device => device.kind === 'audioinput');
            
            return microphones;
        } catch (error) {
            this._reportError(error.message);
            return [];
        }
    }
    
    /**
     * Set active microphone by index
     * @param {number} index - Index of microphone to use
     */
    async setMicrophone(index) {
        const microphones = await this.getMicrophones();
        
        if (index >= 0 && index < microphones.length) {
            this.selectedMicrophoneIndex = index;
            this._reportStatus(`Microphone set to: ${microphones[index].label || 'Device ' + index}`);
        } else {
            this._reportError(`Invalid microphone index: ${index}`);
        }
    }

    /**
     * Start voice recognition
     * @returns {Promise<Object|null>} - Promise resolving to recognition result or null on error
     */
    async startRecognition() {
        if (this.isListening) {
            this._reportStatus('Already listening');
            return null;
        }
        
        try {
            this.isListening = true;
            this._reportStatus('Starting voice recognition...');
            
            // In browser environments, always try Web Speech API first
            if (typeof window !== 'undefined' && 
                (window.SpeechRecognition || window.webkitSpeechRecognition)) {
                return this._startBrowserRecognition();
            } 
            // Fall back to server-side recognition
            else {
                return this._startServerRecognition();
            }
        } catch (error) {
            this.isListening = false;
            this._reportError(error.message || 'Failed to start recognition');
            return null;
        }
    }

    /**
     * Start browser-based speech recognition using Web Speech API
     * @returns {Promise<Object>} - Promise resolving to recognition result
     * @private
     */
    _startBrowserRecognition() {
        return new Promise((resolve, reject) => {
            try {
                // Initialize Speech Recognition API
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                
                recognition.onstart = () => {
                    this._reportStatus('Browser recognition started');
                };
                
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    const confidence = event.results[0][0].confidence;
                    
                    const result = {
                        text: transcript,
                        confidence: confidence
                    };
                    
                    if (this.onRecognitionResult) {
                        this.onRecognitionResult(result);
                    }
                    
                    this._reportStatus('Recognition complete');
                    this.isListening = false;
                    resolve(result);
                };
                
                recognition.onerror = (event) => {
                    this._reportError(`Recognition error: ${event.error}`);
                    this.isListening = false;
                    reject(new Error(event.error));
                };
                
                recognition.onend = () => {
                    this.isListening = false;
                    this._reportStatus('Recognition ended');
                };
                
                // Start recognition
                recognition.start();
                
            } catch (error) {
                this._reportError('Browser recognition failed: ' + error.message);
                reject(error);
            }
        });
    }

    /**
     * Start server-side speech recognition by recording audio and sending to backend
     * @returns {Promise<Object>} - Promise resolving to recognition result
     * @private
     */
    async _startServerRecognition() {
        try {
            // First check if media devices are supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Media recording not supported in this browser');
            }
            
            // Get microphone stream
            const constraints = {
                audio: this.selectedMicrophoneIndex !== null 
                    ? { deviceId: { exact: (await this.getMicrophones())[this.selectedMicrophoneIndex].deviceId } }
                    : true 
            };
            
            this._reportStatus('Requesting microphone access...');
            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.audioChunks = [];
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.currentStream);
            
            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data in 100ms chunks
            this._reportStatus('Recording audio...');
            
            // Stop recording after 5 seconds
            setTimeout(() => {
                if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                }
            }, 5000);
            
            // Return promise that resolves when recording is processed
            return new Promise((resolve, reject) => {
                this.mediaRecorder.onstop = async () => {
                    try {
                        // Stop all tracks in the stream
                        if (this.currentStream) {
                            this.currentStream.getTracks().forEach(track => track.stop());
                            this.currentStream = null;
                        }
                        
                        this._reportStatus('Processing audio...');
                        
                        // Create audio blob from recorded chunks
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                        
                        // Create form data
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'recording.webm');
                        
                        // Send to server for processing
                        const response = await fetch(this.apiBaseUrl, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                        });
                        
                        if (!response.ok) {
                            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                        }
                        
                        // Process result
                        const result = await response.json();
                        
                        // Call callback with result
                        if (this.onRecognitionResult) {
                            this.onRecognitionResult(result);
                        }
                        
                        this._reportStatus('Recognition complete');
                        this.isListening = false;
                        resolve(result);
                    } catch (error) {
                        this._reportError(`Processing error: ${error.message}`);
                        this.isListening = false;
                        reject(error);
                    }
                };
            });
        } catch (error) {
            this.isListening = false;
            this._reportError(`Server recognition failed: ${error.message}`);
            throw error;
        }
    }
    
    /**
     * Stop listening if currently active
     */
    stopListening() {
        if (!this.isListening) {
            return;
        }
        
        this._reportStatus('Stopping recognition...');
        
        // If using media recorder
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        // Stop any active streams
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        
        this.isListening = false;
        this._reportStatus('Recognition stopped by user');
    }
}

// Expose the client globally
window.VoiceRecognitionClient = VoiceRecognitionClient;
console.log("VoiceRecognitionClient class registered globally");

// Create a global instance for easy access
// Try to get the API URL from environment variables if available
const apiBaseUrl = (window.VITE_API_URL && window.VITE_API_URL !== '%VITE_API_URL%') 
    ? `${window.VITE_API_URL}/voice-recognition`
    : '/api/voice-recognition';
    
window.voiceClient = new VoiceRecognitionClient(apiBaseUrl);
console.log("voiceClient instance created globally with API URL:", apiBaseUrl); 