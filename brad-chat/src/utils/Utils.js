
const handleSessionStorage = (session) => {
    console.log("updating sessions", session)
    let current_session = localStorage.getItem('current-session')
    console.log("updating sessions", current_session)
    if (current_session != session){
    localStorage.setItem('current-session', session)
    }
}

export default handleSessionStorage;