import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface IntroOverlayProps {
  onComplete: () => void
}

type Phase = 'idle' | 'en-in' | 'en-visible' | 'en-out' | 'zh-in' | 'zh-visible' | 'zh-out' | 'done'

const FADE_DURATION = 1.2
const HOLD_DURATION = 2.5
const WASH_DURATION = 1.0

export function IntroOverlay({ onComplete }: IntroOverlayProps) {
  const [phase, setPhase] = useState<Phase>('idle')

  useEffect(() => {
    const t = setTimeout(() => setPhase('en-in'), 600)
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    if (phase === 'done') onComplete()
  }, [phase, onComplete])

  const handleEnAnimationComplete = useCallback(() => {
    if (phase === 'en-in') {
      setPhase('en-visible')
      setTimeout(() => setPhase('en-out'), HOLD_DURATION * 1000)
    }
  }, [phase])

  const handleEnExitComplete = useCallback(() => {
    if (phase === 'en-out') {
      setPhase('zh-in')
    }
  }, [phase])

  const handleZhAnimationComplete = useCallback(() => {
    if (phase === 'zh-in') {
      setPhase('zh-visible')
      setTimeout(() => setPhase('zh-out'), HOLD_DURATION * 1000)
    }
  }, [phase])

  const handleZhExitComplete = useCallback(() => {
    if (phase === 'zh-out') {
      setPhase('done')
    }
  }, [phase])

  const showEn = phase === 'en-in' || phase === 'en-visible'
  const showZh = phase === 'zh-in' || phase === 'zh-visible'

  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      pointerEvents: 'none',
      zIndex: 10,
    }}>
      <AnimatePresence onExitComplete={handleEnExitComplete}>
        {showEn && (
          <motion.h1
            key="en"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{
              opacity: 0,
              y: -15,
              filter: 'blur(8px)',
              transition: { duration: WASH_DURATION, ease: 'easeIn' },
            }}
            transition={{ duration: FADE_DURATION, ease: 'easeOut' }}
            onAnimationComplete={handleEnAnimationComplete}
            style={textStyle}
          >
            Hi, I am Mengyu Han
          </motion.h1>
        )}
      </AnimatePresence>

      <AnimatePresence onExitComplete={handleZhExitComplete}>
        {showZh && (
          <motion.h1
            key="zh"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{
              opacity: 0,
              y: -15,
              filter: 'blur(8px)',
              transition: { duration: WASH_DURATION, ease: 'easeIn' },
            }}
            transition={{ duration: FADE_DURATION, ease: 'easeOut' }}
            onAnimationComplete={handleZhAnimationComplete}
            style={{ ...textStyle, fontFamily: "'Noto Sans SC', sans-serif" }}
          >
            你好，我是韩梦宇
          </motion.h1>
        )}
      </AnimatePresence>
    </div>
  )
}

const textStyle: React.CSSProperties = {
  position: 'absolute',
  color: 'white',
  fontFamily: "'Inter', sans-serif",
  fontWeight: 200,
  fontSize: 'clamp(1.8rem, 5vw, 4rem)',
  letterSpacing: '0.05em',
  textShadow: '0 2px 20px rgba(0,0,0,0.3)',
  userSelect: 'none',
}
