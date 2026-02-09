/**
 * Voice Recognition Client
 * 
 * This script provides client-side functionality for interacting with
 * the voice recognition API on the backend.
 */

class VoiceRecognitionClient {
    constructor(apiBaseUrl = '/api/voice') {
        this.apiBaseUrl = apiBaseUrl;
        this.isListening = false;
        this.selectedMicrophoneIndex = null;
        this.onRecognitionResult = null;
        this.onError = null;
        this.onStatusChange = null;
        
        console.log(`Voice Recognition Client initialized with API URL: ${this.apiBaseUrl}`);
        // Test connection on init
        this._testConnection();
    }

    /**
     * Test the connection to the API
     */
    async _testConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/microphones/`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                }
            });
            
            if (response.ok) {
                console.log('Voice Recognition API connection test successful');
            } else {
                console.warn(`Voice Recognition API connection test failed: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.error(`Voice Recognition API connection test error: ${error.message}`);
        }
    }

    /**
     * Set callback for when recognition results are received
     * @param {Function} callback - Function that takes result object as parameter
     */
    setRecognitionResultCallback(callback) {
        this.onRecognitionResult = callback;
    }

    /**
     * Set callback for errors
     * @param {Function} callback - Function that takes error message as parameter
     */
    setErrorCallback(callback) {
        this.onError = callback;
    }

    /**
     * Set callback for status changes
     * @param {Function} callback - Function that takes status message as parameter
     */
    setStatusCallback(callback) {
        this.onStatusChange = callback;
    }

    /**
     * Report status change
     * @param {string} status - Status message
     */
    _reportStatus(status) {
        console.log(`Voice Recognition Status: ${status}`);
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }

    /**
     * Report error
     * @param {string} error - Error message
     */
    _reportError(error) {
        console.error(`Voice Recognition Error: ${error}`);
        if (this.onError) {
            this.onError(error);
        }
    }

    /**
     * Get available microphones
     * @returns {Promise<Array>} List of available microphones
     */
    async getMicrophones() {
        try {
            this._reportStatus('Fetching available microphones...');
            console.log(`Fetching from: ${this.apiBaseUrl}/microphones/`);
            
            const response = await fetch(`${this.apiBaseUrl}/microphones/`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                }
            });
            
            console.log('Microphones response status:', response.status);
            console.log('Microphones response headers:', [...response.headers.entries()]);
            
            if (!response.ok) {
                let errorMessage = `Server returned ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    errorMessage += ' (Failed to parse error response)';
                }
                throw new Error(errorMessage);
            }
            
            const data = await response.json();
            console.log('Microphones data:', data);
            this._reportStatus(`Found ${data.microphones.length} microphones`);
            return data.microphones;
            
        } catch (error) {
            this._reportError(`Failed to get microphones: ${error.message}`);
            return [];
        }
    }

    /**
     * Set microphone to use for voice recognition
     * @param {number} index - Index of the microphone to use
     */
    setMicrophone(index) {
        this.selectedMicrophoneIndex = index;
        this._reportStatus(`Microphone set to index ${index}`);
    }

    /**
     * Start voice recognition using selected or default microphone
     */
    async startRecognition() {
        if (this.isListening) {
            this._reportStatus('Already listening');
            return;
        }

        try {
            this.isListening = true;
            this._reportStatus('Starting voice recognition...');
            
            // Prepare request data
            const requestData = new FormData();
            if (this.selectedMicrophoneIndex !== null) {
                requestData.append('microphone_index', this.selectedMicrophoneIndex);
            }
            
            console.log(`Starting recognition at: ${this.apiBaseUrl}/recognize/`);
            
            // Make API call
            const response = await fetch(`${this.apiBaseUrl}/recognize/`, {
                method: 'POST',
                body: requestData,
                headers: {
                    'Accept': 'application/json',
                }
            });
            
            // Reset listening state
            this.isListening = false;
            
            console.log('Recognition response status:', response.status);
            console.log('Recognition response headers:', [...response.headers.entries()]);
            
            if (!response.ok) {
                let errorMessage = `Server returned ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    errorMessage += ' (Failed to parse error response)';
                }
                throw new Error(errorMessage);
            }
            
            // Process result
            const result = await response.json();
            console.log('Recognition result:', result);
            this._reportStatus('Recognition complete');
            
            // Invoke callback with results
            if (this.onRecognitionResult) {
                this.onRecognitionResult(result);
            }
            
            return result;
            
        } catch (error) {
            this.isListening = false;
            this._reportError(`Recognition failed: ${error.message}`);
            return null;
        }
    }
} 