import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import AuthModal from './components/AuthModal';
import ChatPage from './pages/ChatPage';
import APIKeysPage from './pages/APIKeysPage';
import UpgradePage from './pages/UpgradePage';
import ContactPage from './pages/ContactPage';
import DocsPage from './pages/DocsPage';
import PaymentPage from './pages/PaymentPage';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [view, setView] = useState('chat'); // 'chat', 'dev', or 'upgrade'

  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [docId, setDocId] = useState(null);
  const [guestId, setGuestId] = useState(null);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'Hello! Upload a PDF, TXT, or Word file to start chatting.' }
  ]);
  const [loading, setLoading] = useState(false);
  const [usage, setUsage] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [confirmRegenId, setConfirmRegenId] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user_id');
    const storedEmail = localStorage.getItem('email');
    if (token && storedUser) {
      setUser({ token, id: storedUser, email: storedEmail });
      fetchUsage(token);
      fetchDocuments(token);
    }
  }, []);

  // Context Switch Notification & History Fetch
  useEffect(() => {
    const fetchHistory = async () => {
      if (docId) {
        // Clear current view
        setMessages([]);

        try {
          // 1. Fetch History from DB
          const headers = user ? { Authorization: `Bearer ${user.token}` } : {};
          const resp = await axios.get(`${API_BASE}/documents/${docId}/history`, { headers });

          if (resp.data && resp.data.length > 0) {
            setMessages(resp.data);
          } else {
            // 2. Or show fresh start message
            const activeDoc = documents.find(d => d.id === docId);
            const name = activeDoc ? activeDoc.name : (file ? file.name : "Selected Model");
            setMessages([{
              role: 'system',
              content: `Switched to: ${name}.`
            }]);
          }
        } catch (e) {
          console.error("Failed to fetch chat history", e);
        }
      }
    };

    fetchHistory();
  }, [docId, user?.token]);

  const fetchDocuments = async (token) => {
    try {
      const resp = await axios.get(`${API_BASE}/documents`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDocuments(resp.data);
    } catch (e) {
      console.error("Failed to fetch documents", e);
    }
  };

  const fetchUsage = async (token) => {
    try {
      const resp = await axios.get(`${API_BASE}/usage`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUsage(resp.data);
    } catch (e) {
      console.error("Failed to fetch usage", e);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    try {
      const endpoint = authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
      const response = await axios.post(`${API_BASE}${endpoint}`, { email, password });

      if (authMode === 'login') {
        const userData = { token: response.data.access_token, id: response.data.user_id, email: email };
        localStorage.setItem('token', userData.token);
        localStorage.setItem('user_id', userData.id);
        localStorage.setItem('email', email);
        setUser(userData);
        setShowAuth(false);
        fetchUsage(userData.token);
        fetchDocuments(userData.token);
      } else {
        alert("Registered! Please login.");
        setAuthMode('login');
      }
    } catch (error) {
      alert(error.response?.data?.detail || "Auth failed");
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user_id');
    localStorage.removeItem('email');
    setUser(null);
    setDocId(null);
    setUsage(null);
    setDocuments([]);
  };

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const headers = user ? { Authorization: `Bearer ${user.token}` } : {};
      const response = await axios.post(`${API_BASE}/upload`, formData, { headers });
      setDocId(response.data.doc_id);
      if (!user) setGuestId(response.data.user_id);
      setFile(selectedFile);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Processed "${selectedFile.name}". ${!user ? 'You have 3 free queries.' : ''}`
      }]);
      if (user) {
        fetchUsage(user.token);
        fetchDocuments(user.token);
      }
    } catch (error) {
      alert('Upload failed. Please check connection.');
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || !docId || loading) return;

    const userMsg = query;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setQuery('');
    setLoading(true);

    try {
      const headers = user ? { Authorization: `Bearer ${user.token}` } : {};
      const params = { doc_id: docId, query: userMsg };
      if (!user && guestId) params.guest_id = guestId;

      const response = await axios.post(`${API_BASE}/query`, null, {
        params,
        headers
      });

      const remaining = response.data.queries_remaining;
      let botMsg = response.data.answer;
      if (user === null && remaining !== undefined) {
        botMsg += `\n\n(Guest limit: ${remaining} queries remaining)`;
      }

      setMessages(prev => [...prev, { role: 'system', content: botMsg }]);
      if (user) fetchUsage(user.token);
    } catch (error) {
      if (error.response?.status === 403 || error.response?.status === 401) {
        setMessages(prev => [...prev, {
          role: 'system',
          content: 'Free trial limit reached! Your guest data has been purged. Please login and re-upload to continue without limits.'
        }]);
        setDocId(null);
        setGuestId(null);
        setFile(null);
        setShowAuth(true);
      } else {
        setMessages(prev => [...prev, { role: 'system', content: 'Connection or API key error.' }]);
      }
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (id) => {
    if (!window.confirm("Delete this document and context?")) return;
    try {
      await axios.delete(`${API_BASE}/documents/${id}`, {
        headers: { Authorization: `Bearer ${user.token}` }
      });
      fetchDocuments(user.token);
      fetchUsage(user.token);
      if (docId === id) {
        setDocId(null);
        setFile(null);
      }
    } catch (e) {
      alert("Delete failed");
    }
  };

  const generateKey = async (id) => {
    try {
      const resp = await axios.post(`${API_BASE}/documents/${id}/api-key`, null, {
        headers: { Authorization: `Bearer ${user.token}` }
      });
      alert(`API Key Generated: ${resp.data.api_key}\nKeep it safe!`);
      fetchDocuments(user.token);
    } catch (e) {
      alert("Key generation failed");
    }
  };

  return (
    <div className="app-container">
      <div className="app-wrapper"></div>
      <Navbar
        user={user}
        view={view}
        setView={setView}
        logout={logout}
        setShowAuth={setShowAuth}
      />

      <div className={`main-content ${view !== 'chat' ? 'scrollable' : ''}`}>
        {view === 'chat' ? (
          <ChatPage
            apiBase={API_BASE}
            user={user}
            usage={usage}
            uploading={uploading}
            file={file}
            handleFileUpload={handleFileUpload}
            documents={documents}
            docId={docId}
            setDocId={setDocId}
            setFile={setFile}
            messages={messages}
            setMessages={setMessages}
            loading={loading}
            query={query}
            setQuery={setQuery}
            handleQuery={handleQuery}
            chatEndRef={chatEndRef}
          />
        ) : view === 'dev' ? (
          <APIKeysPage
            documents={documents}
            usage={usage}
            generateKey={generateKey}
            deleteDocument={deleteDocument}
            confirmRegenId={confirmRegenId}
            setConfirmRegenId={setConfirmRegenId}
            showCreateModal={showCreateModal}
            setShowCreateModal={setShowCreateModal}
            selectedDocId={selectedDocId}
            setSelectedDocId={setSelectedDocId}
          />
        ) : view === 'upgrade' ? (
          <UpgradePage setView={setView} user={user} setShowAuth={setShowAuth} />
        ) : view === 'contact' ? (
          <ContactPage setView={setView} />
        ) : view === 'payment' ? (
          <PaymentPage setView={setView} user={user} />
        ) : (
          <DocsPage setView={setView} user={user} setShowAuth={setShowAuth} />
        )}
      </div>

      <AuthModal
        showAuth={showAuth}
        setShowAuth={setShowAuth}
        authMode={authMode}
        setAuthMode={setAuthMode}
        email={email}
        setEmail={setEmail}
        password={password}
        setPassword={setPassword}
        handleAuth={handleAuth}
      />
    </div>
  );
}

export default App;
