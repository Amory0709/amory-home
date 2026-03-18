import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export function ExploreHint() {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const hide = () => setVisible(false)

    window.addEventListener('pointerdown', hide, { once: true })
    window.addEventListener('wheel', hide, { once: true })
    window.addEventListener('touchstart', hide, { once: true })

    return () => {
      window.removeEventListener('pointerdown', hide)
      window.removeEventListener('wheel', hide)
      window.removeEventListener('touchstart', hide)
    }
  }, [])

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.8 }}
          style={{
            position: 'absolute',
            bottom: '2.5rem',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <span style={{
            color: 'rgba(255,255,255,0.6)',
            fontFamily: "'Inter', sans-serif",
            fontWeight: 300,
            fontSize: '0.85rem',
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
          }}>
            Drag to explore
          </span>
          <motion.svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          >
            <path
              d="M4 7L10 13L16 7"
              stroke="rgba(255,255,255,0.5)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </motion.svg>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
