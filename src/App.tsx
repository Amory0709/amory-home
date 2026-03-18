import { useEffect, useRef, useState, useCallback } from 'react'
import { AnimatePresence } from 'framer-motion'
import { SceneManager } from './webgl/SceneManager'
import { IntroOverlay } from './components/IntroOverlay'
import { ExploreHint } from './components/ExploreHint'
import { DrawingToolbar } from './components/DrawingToolbar'
import './App.css'

const LONG_PRESS_MS = 500
const LONG_PRESS_MOVE_THRESHOLD = 8

function App() {
  const containerRef = useRef<HTMLDivElement>(null)
  const managerRef = useRef<SceneManager | null>(null)
  const [introDone, setIntroDone] = useState(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(3)

  useEffect(() => {
    if (!containerRef.current) return
    managerRef.current = new SceneManager(containerRef.current)
    return () => {
      managerRef.current?.dispose()
      managerRef.current = null
    }
  }, [])

  const handleIntroComplete = useCallback(() => {
    setIntroDone(true)
  }, [])

  // Long-press detection to enter drawing mode
  useEffect(() => {
    const mgr = managerRef.current
    if (!introDone || isDrawing || !mgr) return

    const canvas = mgr.getDomElement()
    let timer: number | null = null
    let startX = 0, startY = 0
    let latestX = 0, latestY = 0

    const onDown = (e: PointerEvent) => {
      startX = latestX = e.clientX
      startY = latestY = e.clientY
      timer = window.setTimeout(() => {
        setIsDrawing(true)
        mgr.setDrawingMode(true)
        mgr.startStrokeFromLongPress(latestX, latestY)
        timer = null
      }, LONG_PRESS_MS)
    }

    const onMove = (e: PointerEvent) => {
      latestX = e.clientX
      latestY = e.clientY
      if (timer !== null) {
        const dx = e.clientX - startX
        const dy = e.clientY - startY
        if (Math.sqrt(dx * dx + dy * dy) > LONG_PRESS_MOVE_THRESHOLD) {
          clearTimeout(timer)
          timer = null
        }
      }
    }

    const onUp = () => {
      if (timer !== null) {
        clearTimeout(timer)
        timer = null
      }
    }

    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointercancel', onUp)

    return () => {
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      if (timer !== null) clearTimeout(timer)
    }
  }, [introDone, isDrawing])

  // Sync brush settings to SceneManager
  useEffect(() => {
    managerRef.current?.setBrushSize(brushSize)
  }, [brushSize])

  const handleCloseDrawing = useCallback(() => {
    setIsDrawing(false)
    managerRef.current?.setDrawingMode(false)
  }, [])

  const handleUndo = useCallback(() => {
    managerRef.current?.undoDrawing()
  }, [])

  const handleClear = useCallback(() => {
    managerRef.current?.clearDrawing()
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden', position: 'relative' }}>
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}
      />

      {!introDone && <IntroOverlay onComplete={handleIntroComplete} />}
      {introDone && !isDrawing && <ExploreHint />}

      <AnimatePresence>
        {isDrawing && (
          <DrawingToolbar
            brushSize={brushSize}
            onBrushSizeChange={setBrushSize}
            onUndo={handleUndo}
            onClear={handleClear}
            onClose={handleCloseDrawing}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
