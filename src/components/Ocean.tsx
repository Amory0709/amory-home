import { useRef, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { OrbitControls } from '@react-three/drei'

export function Ocean() {
  const meshRef = useRef<THREE.Mesh>(null)
  const { viewport, camera } = useThree()

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uResolution: { value: new THREE.Vector2(viewport.width, viewport.height) },
      uCameraPos: { value: new THREE.Vector3() },
      uCameraDir: { value: new THREE.Vector3() },
      uCameraUp: { value: new THREE.Vector3() },
      uCameraRight: { value: new THREE.Vector3() },
    }),
    [viewport]
  )

  useFrame((state) => {
    if (!meshRef.current) return
    
    uniforms.uTime.value = state.clock.elapsedTime
    uniforms.uResolution.value.set(viewport.width, viewport.height)
    
    // 传递相机信息给 shader
    uniforms.uCameraPos.value.copy(camera.position)
    
    // 计算相机方向向量
    const lookAt = new THREE.Vector3(0, 0, 0)
    const cf = new THREE.Vector3().subVectors(lookAt, camera.position).normalize()
    const cr = new THREE.Vector3().crossVectors(cf, new THREE.Vector3(0, 1, 0)).normalize()
    const cu = new THREE.Vector3().crossVectors(cr, cf).normalize()
    
    uniforms.uCameraDir.value.copy(cf)
    uniforms.uCameraRight.value.copy(cr)
    uniforms.uCameraUp.value.copy(cu)
  })

  return (
    <>
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={0.5}
        maxDistance={10}
        target={[0, 0, 0]}
        camera={camera}
      />
      
      {/* 全屏 quad */}
      <mesh ref={meshRef}>
        <planeGeometry args={[2, 2]} />
        <shaderMaterial
          uniforms={uniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </>
  )
}

// 简单的 vertex shader，传递 UV 到全屏
const vertexShader = /* glsl */ `
  varying vec2 vUv;
  
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

const fragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec3 uCameraPos;
  uniform vec3 uCameraDir;
  uniform vec3 uCameraRight;
  uniform vec3 uCameraUp;
  
  varying vec2 vUv;
  
  // ========== 从 common.gl 移植的代码 ==========
  
  const float PI = 3.1415927;
  
  // RAYMARCHING
  const int STEPS = 200;
  const int STEPS_GROUND = 50;
  
  // WATER PARAMS
  const int numWaves = 60;
  const float oceanHeight = 0.2;
  const float waveBaseHeight = 0.5;
  const float waveMaxAmplitude = 0.35;
  
  // 修改水的颜色为 tiffany 蓝
  const vec3 waterCol = vec3(0.5, 0.85, 0.82); // tiffany 蓝
  const float waterAbsorp = 0.7;
  const vec3 subsurfCol = waterCol * vec3(1.3, 1.5, 1.1);
  
  // MATS & LIGHT
  const int MAT_OCEAN = 0;
  const int MAT_GROUND = 1;
  const vec3 ld = normalize(vec3(-1.0, -1.0, -2.0));
  
  float saturate(float v) { return clamp(v, 0.0, 1.0); }
  
  // Hash function
  uint murmurHash11(uint src) {
    const uint M = 0x5bd1e995u;
    uint h = 1190494759u;
    src *= M; src ^= src >> 24u; src *= M;
    h *= M; h ^= src;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
  }
  
  float hash11(float src) {
    uint h = murmurHash11(floatBitsToUint(src));
    return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
  }
  
  float SingleWaveHeight(vec2 uv, vec2 dir, float speed, float ampl, float time) {
    float d = dot(uv, dir);
    float ph = d * 10.0 + time * speed;
    float h = (sin(ph) * 0.5 + 0.5);
    h = pow(h, 2.0);
    h = h * 2.0 - 1.0;
    return h * ampl;
  }
  
  float WaveHeight(vec2 uv, float time, int num) {
    uv *= 1.6;
    
    float h = 0.0;
    float w = 1.0;
    float tw = 0.0;
    float s = 1.0;
    const float phBase = 0.2;
    
    for(int i = 0; i < 60; i++) {
      if(i >= num) break;
      
      float rand = hash11(float(i)) * 2.0 - 1.0;
      float dirMaxDiffer = float(i) / float(numWaves - 1);
      dirMaxDiffer = pow(dirMaxDiffer, 1.0) * 2.0 * PI;
      float ph = phBase + rand * 0.75 * PI;
      vec2 dir = vec2(sin(ph), cos(ph));
      h += SingleWaveHeight(uv, dir, 1.0 + s * 0.05, w, time);
      tw += w;
      const float scale = 1.0812;
      w /= scale;
      uv *= scale;
      s *= scale;
    }
    
    h /= tw;
    h = waveBaseHeight + waveMaxAmplitude * h;
    return h;
  }
  
  // Simplex noise (从 common.gl)
  vec4 mod289(vec4 x) {
    return x - floor(x / 289.0) * 289.0;
  }
  
  vec4 permute(vec4 x) {
    return mod289((x * 34.0 + 1.0) * x);
  }
  
  float causticNoiseBlur;
  
  vec4 snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    
    vec3 i = floor(v + dot(v, vec3(C.y)));
    vec3 x0 = v - i + dot(i, vec3(C.x));
    
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    
    vec3 x1 = x0 - i1 + C.x;
    vec3 x2 = x0 - i2 + C.y;
    vec3 x3 = x0 - 0.5;
    
    vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
      i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
      i.x + vec4(0.0, i1.x, i2.x, 1.0));
    
    vec4 j = p - 49.0 * floor(p / 49.0);
    
    vec4 x_ = floor(j / 7.0);
    vec4 y_ = floor(j - 7.0 * x_);
    
    vec4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    vec4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;
    
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    
    vec3 g0 = vec3(a0.xy, h.x);
    vec3 g1 = vec3(a0.zw, h.y);
    vec3 g2 = vec3(a1.xy, h.z);
    vec3 g3 = vec3(a1.zw, h.w);
    
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    vec4 m2 = m * m;
    vec4 m3 = m2 * m;
    vec4 m4 = m2 * m2;
    
    vec3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    
    vec4 px = vec4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return mix(42.0, 0.0, causticNoiseBlur) * vec4(grad, dot(m4, px));
  }
  
  // ========== 从 image.gl 移植的代码 ==========
  
  float WaterHeight(vec3 p, int waveCount) {
    float h = WaveHeight(p.xz * 0.1, uTime, waveCount) + oceanHeight;
    return h;
  }
  
  float GroundHeight(vec3 p) {
    float h = 0.0;
    float tw = 0.0;
    float w = 1.0;
    p *= 0.2;
    p.xz += vec2(-1.25, 0.35);
    
    for(int i = 0; i < 2; i++) {
      h += w * sin(p.x) * sin(p.z);
      const float s = 1.173;
      tw += w;
      p *= s;
      p.xz += vec2(2.373, 0.977);
      w /= s;
    }
    
    h /= tw;
    float hGround = -0.2 + 1.65 * h;
    return hGround;
  }
  
  float sdOcean(vec3 p) {
    float dh = p.y - WaterHeight(p, numWaves);
    dh *= 0.75;
    return dh;
  }
  
  float sdOcean_Levels(vec3 p, int waveCount) {
    float dh = p.y - WaterHeight(p, waveCount);
    dh *= 0.75;
    return dh;
  }
  
  int material;
  
  float map(vec3 p, bool includeWater) {
    float hGround = GroundHeight(p);
    float dGround = p.y - hGround;
    dGround *= 0.9;
    float d = dGround;
    
    material = MAT_GROUND;
    if(includeWater) {
      float dOcean = sdOcean(p);
      material = dOcean < d ? MAT_OCEAN : material;
      d = min(d, dOcean);
    }
    return d;
  }
  
  float RM(vec3 ro, vec3 rd) {
    float t = 0.0;
    float s = 1.0;
    for(int i = 0; i < STEPS; i++) {
      float d = map(ro + t * rd, true);
      if(d < 0.001) return t;
      t += d * s;
      s *= 1.02;
    }
    return -t;
  }
  
  float RM_Ground(vec3 ro, vec3 rd) {
    float t = 0.0;
    for(int i = 0; i < STEPS_GROUND; i++) {
      float d = map(ro + t * rd, false);
      if(d < 0.001) return t;
      t += d;
    }
    return -t;
  }
  
  vec3 Normal(vec3 p) {
    const float h = 0.001;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
      k.xyy * map(p + k.xyy * h, true) + 
      k.yyx * map(p + k.yyx * h, true) + 
      k.yxy * map(p + k.yxy * h, true) + 
      k.xxx * map(p + k.xxx * h, true)
    );
  }
  
  vec3 WaveNormal_Levels(vec3 p, int levels) {
    const float h = 0.001;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
      k.xyy * sdOcean_Levels(p + k.xyy * h, levels) + 
      k.yyx * sdOcean_Levels(p + k.yyx * h, levels) + 
      k.yxy * sdOcean_Levels(p + k.yxy * h, levels) + 
      k.xxx * sdOcean_Levels(p + k.xxx * h, levels)
    );
  }
  
  float water_caustics(vec3 pos) {
    vec4 n = snoise(pos);
    
    pos -= 0.07 * n.xyz;
    pos *= 1.62;
    n = snoise(pos);
    
    pos -= 0.07 * n.xyz;
    n = snoise(pos);
    
    pos -= 0.07 * n.xyz;
    n = snoise(pos);
    
    return n.w;
  }
  
  void DarkenGround(inout vec3 col, vec3 groundPos, float oceanHeight, out float wetness) {
    wetness = 1.0 - smoothstep(0.05, 0.2, groundPos.y - oceanHeight - 0.3);
    col = mix(col, col * vec3(0.95, 0.92, 0.85) * 0.8, wetness);
  }
  
  vec3 Reflection(vec3 refl, float fresnel) {
    float spec = max(0.0, dot(refl, -ld));
    spec = pow(spec, 256.0);
    vec3 col = spec * vec3(1.0);
    
    // 简单的天空颜色替代 iChannel0 cubemap
    vec3 skyColor = vec3(0.4, 0.7, 0.95);
    skyColor = mix(skyColor, vec3(1.0), max(0.0, (1.0 - refl.y) * 0.3));
    col += fresnel * skyColor * 0.4;
    
    return col;
  }
  
  float Fresnel(vec3 rd, vec3 nor) {
    float fresnel = 1.0 - abs(dot(nor, rd));
    fresnel = pow(fresnel, 6.0);
    return fresnel;
  }
  
  vec3 Render(float t, vec3 ro, vec3 rd) {
    // 没有击中 -> 渲染背景
    if(t < 0.0) {
      vec3 col = vec3(0.35, 0.62, 0.9);
      col = mix(col, vec3(1.0), max(0.0, (1.0 - rd.y) * 0.3));
      float sunDot = max(0.0, dot(rd, -ld));
      sunDot = pow(sunDot, 6.0);
      sunDot = tanh(sunDot);
      col += sunDot * vec3(1.0, 0.8, 0.7);
      return col;
    }
    
    vec3 p = ro + t * rd;
    vec3 refl;
    vec3 pGround;
    const vec3 groundCol = vec3(0.9, 0.85, 0.7);
    vec3 col = groundCol;
    vec3 transmittance = vec3(1.0);
    
    if(material == MAT_OCEAN) {
      float hGround = GroundHeight(p);
      float dGround = p.y - hGround;
      float nearShoreAlpha = 1.0 - smoothstep(0.5, -0.2, hGround - oceanHeight);
      
      vec3 nor = Normal(p);
      nor = normalize(mix(nor, vec3(0.0, 1.0, 0.0), nearShoreAlpha * 0.9));
      refl = reflect(rd, nor);
      vec3 refr = refract(rd, nor, 1.0 / 1.2);
      if(refr == vec3(0.0)) refr = refl;
      
      float tGround = RM_Ground(p, refr);
      if(tGround < 0.0) tGround = 4.0;
      pGround = p + tGround * refr;
      
      float fresnel = Fresnel(rd, nor);
      vec3 norSubsurf = WaveNormal_Levels(p, numWaves / 3);
      const vec3 ldSubsurf = ld * vec3(1.0, -1.0, 1.0);
      float subsurf = max(0.0, max(0.0, dot(rd, -ldSubsurf)) * dot(norSubsurf, ldSubsurf));
      subsurf = pow(subsurf, 2.0);
      subsurf *= 1.0 - fresnel;
      subsurf *= 0.5;
      
      float wetness;
      DarkenGround(col, pGround, oceanHeight, wetness);
      
      float spec = max(0.0, dot(refl, -ld));
      spec = pow(spec, 256.0);
      
      transmittance = exp(-tGround * waterAbsorp / waterCol);
      float waterAlpha = 1.0 - exp(-tGround * waterAbsorp * 0.5);
      
      vec3 causticPos = pGround * 2.0 + vec3(0.0, uTime * 0.15, 0.0);
      float causticAlpha = 1.0 - saturate(exp(-tGround * 2.0));
      causticNoiseBlur = 1.0 - min(1.0, causticAlpha * 2.0);
      vec3 o = vec3(1.0, 0.0, 1.0) * 0.02;
      vec3 caustics;
      caustics.x = mix(water_caustics(causticPos + o), water_caustics(causticPos + o + 1.0), 0.5);
      caustics.y = mix(water_caustics(causticPos + o * 4.0), water_caustics(causticPos + o + 1.0), 0.5);
      caustics.z = mix(water_caustics(causticPos + o * 6.0), water_caustics(causticPos + o + 1.0), 0.5);
      caustics = exp(caustics * 4.0 - 1.0);
      caustics *= causticAlpha;
      col += caustics;
      
      col *= transmittance;
      col += tGround * exp(-tGround * waterAbsorp) * waterCol * 0.3;
      col += subsurf * subsurfCol;
      col += Reflection(refl, fresnel);
    }
    else if(material == MAT_GROUND) {
      pGround = p;
      float wetness;
      DarkenGround(col, pGround, oceanHeight, wetness);
      
      vec3 nor = Normal(p);
      vec3 refl = reflect(rd, nor);
      float fresnel = Fresnel(rd, nor);
      col += wetness * Reflection(refl, fresnel);
    }
    
    return col;
  }
  
  void main() {
    // 将 vUv 从 [0,1] 转换为 NDC 坐标 [-1,1]
    vec2 uv = vUv * 2.0 - 1.0;
    
    // 使用传入的相机参数构建光线
    vec3 ro = uCameraPos;
    const float fl = 1.0; // focal length
    vec3 rd = normalize(uv.x * uCameraRight + uv.y * uCameraUp + fl * uCameraDir);
    
    float d = RM(ro, rd);
    vec3 col = Render(d, ro, rd);
    
    // Gamma 校正
    col = pow(col, vec3(1.0 / 2.2));
    
    gl_FragColor = vec4(col, 1.0);
  }
`
