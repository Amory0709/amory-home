import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function Ocean() {
  const meshRef = useRef<THREE.Mesh>(null)

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uDeepColor: { value: new THREE.Color('#006994') },
      uShallowColor: { value: new THREE.Color('#48D1CC') },
    }),
    []
  )

  useFrame((_, delta) => {
    uniforms.uTime.value += delta
  })

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <planeGeometry args={[30, 30, 128, 128]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        transparent
      />
    </mesh>
  )
}

const vertexShader = /* glsl */ `
  uniform float uTime;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    vUv = uv;

    vec3 pos = position;

    // Gentle waves
    float wave1 = sin(pos.x * 0.5 + uTime * 0.8) * 0.15;
    float wave2 = sin(pos.y * 0.3 + uTime * 0.6) * 0.1;
    float wave3 = sin((pos.x + pos.y) * 0.7 + uTime * 1.2) * 0.05;

    pos.z += wave1 + wave2 + wave3;
    vElevation = pos.z;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`

const fragmentShader = /* glsl */ `
  uniform vec3 uDeepColor;
  uniform vec3 uShallowColor;
  uniform float uTime;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    // Mix colors based on wave height
    float mixFactor = smoothstep(-0.1, 0.2, vElevation);
    vec3 color = mix(uDeepColor, uShallowColor, mixFactor);

    // Subtle caustic shimmer
    float caustic = sin(vUv.x * 20.0 + uTime) * sin(vUv.y * 20.0 + uTime * 0.7);
    caustic = smoothstep(0.3, 1.0, caustic) * 0.15;
    color += caustic;

    gl_FragColor = vec4(color, 0.9);
  }
`
