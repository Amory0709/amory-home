import { useEffect, useRef } from 'react'
import { SceneManager } from './webgl/SceneManager'
import './App.css'

function App() {
  const containerRef = useRef<HTMLDivElement>(null)
  const managerRef = useRef<SceneManager | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    // Initialize the Three.js scene manager
    managerRef.current = new SceneManager(containerRef.current)

    return () => {
      // Cleanup on unmount
      managerRef.current?.dispose()
      managerRef.current = null
    }
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden', position: 'relative' }}>
      <div 
        ref={containerRef} 
        style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} 
      />
      {/* HTML Overlays will go here */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <h1 style={{ color: 'white', fontFamily: 'sans-serif', fontSize: '3rem', opacity: 0, transition: 'opacity 1s ease-in-out' }} id="intro-text">
          Hi, I am Mengyu Han
        </h1>
      </div>
    </div>
  )
}

export default App
