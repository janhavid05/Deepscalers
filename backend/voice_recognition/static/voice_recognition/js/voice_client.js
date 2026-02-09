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
    setStatusChangeCallback(callback) {
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
     * Get available microphones from the server
     * @returns {Promise} - Promise resolving to array of microphone objects
     */
    async getAvailableMicrophones() {
        try {
            this._reportStatus('Fetching available microphones...');
            const response = await fetch(`${this.apiBaseUrl}/microphones/`);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to get microphones: ${errorText}`);
            }
            
            const data = await response.json();
            this._reportStatus(`Found ${data.microphones.length} microphones`);
            return data.microphones;
        } catch (error) {
            this._reportError(`Error fetching microphones: ${error.message}`);
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
            
            // Make API call
            const response = await fetch(`${this.apiBaseUrl}/recognize/`, {
                method: 'POST',
                body: requestData
            });
            
            // Reset listening state
            this.isListening = false;
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Recognition failed');
            }
            
            // Process result
            const result = await response.json();
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

    /**
     * Upload an audio file for transcription
     * @param {File} audioFile - Audio file to transcribe
     * @returns {Promise} - Promise resolving to transcription result
     */
    async transcribeAudioFile(audioFile) {
        try {
            this._reportStatus('Uploading audio file for transcription...');
            
            const formData = new FormData();
            formData.append('audio', audioFile);
            
            const response = await fetch(`${this.apiBaseUrl}/audio-files/transcribe_audio_data/`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Transcription failed');
            }
            
            const result = await response.json();
            this._reportStatus('Transcription complete');
            
            return result;
        } catch (error) {
            this._reportError(`Transcription failed: ${error.message}`);
            return null;
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VoiceRecognitionClient };
} 