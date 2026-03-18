import { useRef, useCallback } from 'react'
import { motion } from 'framer-motion'

interface DrawingToolbarProps {
  brushSize: number
  onBrushSizeChange: (size: number) => void
  onUndo: () => void
  onClear: () => void
  onClose: () => void
}

const MIN_SIZE = 3
const MAX_SIZE = 30
const TRACK_WIDTH = 90

export function DrawingToolbar({
  brushSize,
  onBrushSizeChange,
  onUndo,
  onClear,
  onClose,
}: DrawingToolbarProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const fraction = (brushSize - MIN_SIZE) / (MAX_SIZE - MIN_SIZE)

  const updateFromPointer = useCallback((clientX: number) => {
    const track = trackRef.current
    if (!track) return
    const rect = track.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    onBrushSizeChange(Math.round(MIN_SIZE + ratio * (MAX_SIZE - MIN_SIZE)))
  }, [onBrushSizeChange])

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation()
    dragging.current = true
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    updateFromPointer(e.clientX)
  }, [updateFromPointer])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return
    updateFromPointer(e.clientX)
  }, [updateFromPointer])

  const onPointerUp = useCallback(() => {
    dragging.current = false
  }, [])

  const thumbSize = 6 + fraction * 14

  return (
    <motion.div
      initial={{ y: 60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 60, opacity: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      style={toolbarStyle}
      onPointerDown={e => e.stopPropagation()}
    >
      {/* Custom slider */}
      <div style={sectionStyle}>
        <div
          ref={trackRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          style={{
            width: TRACK_WIDTH,
            height: 28,
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            touchAction: 'none',
            position: 'relative',
          }}
        >
          {/* Track background */}
          <div style={{
            position: 'absolute',
            left: 0,
            right: 0,
            height: 3,
            borderRadius: 2,
            background: 'rgba(255,255,255,0.15)',
          }} />
          {/* Active portion */}
          <div style={{
            position: 'absolute',
            left: 0,
            width: `${fraction * 100}%`,
            height: 3,
            borderRadius: 2,
            background: 'linear-gradient(90deg, rgba(217,166,179,0.4), rgba(217,166,179,0.85))',
          }} />
          {/* Thumb */}
          <div style={{
            position: 'absolute',
            left: `calc(${fraction * 100}% - ${thumbSize / 2}px)`,
            width: thumbSize,
            height: thumbSize,
            borderRadius: '50%',
            background: 'rgba(217,166,179,0.85)',
            boxShadow: '0 0 8px rgba(217,166,179,0.5)',
            transition: dragging.current ? 'none' : 'width 0.15s, height 0.15s, left 0.15s',
          }} />
        </div>
      </div>

      <div style={divider} />

      <div style={{ ...sectionStyle, gap: '4px' }}>
        <button onClick={onUndo} style={btnStyle} title="Undo">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 7v6h6" /><path d="M3 13a9 9 0 0 1 15.36-6.36" />
          </svg>
        </button>
        <button onClick={onClear} style={btnStyle} title="Clear">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 6h18" /><path d="M8 6V4h8v2" /><path d="M5 6l1 14h12l1-14" />
          </svg>
        </button>
        <button onClick={onClose} style={{ ...btnStyle, marginLeft: '4px' }} title="Done">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
            <path d="M18 6L6 18" /><path d="M6 6l12 12" />
          </svg>
        </button>
      </div>
    </motion.div>
  )
}

const toolbarStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: '1.5rem',
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '8px 16px',
  borderRadius: '16px',
  background: 'rgba(255,255,255,0.1)',
  backdropFilter: 'blur(24px)',
  WebkitBackdropFilter: 'blur(24px)',
  border: '1px solid rgba(255,255,255,0.15)',
  boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
  zIndex: 20,
  pointerEvents: 'auto',
  userSelect: 'none',
}

const sectionStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
}

const divider: React.CSSProperties = {
  width: 1,
  height: 20,
  background: 'rgba(255,255,255,0.15)',
}

const btnStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.08)',
  border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: '8px',
  width: '32px',
  height: '32px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  transition: 'background 0.2s',
  padding: 0,
}
