import { useEffect, useRef, useState } from 'react';
import { EmbeddingScene, VOCAB_SIZE, HIDDEN_SIZE } from './three/embeddingScene';

export default function App(): React.JSX.Element {
  const canvasRef: React.RefObject<HTMLCanvasElement | null> = useRef<HTMLCanvasElement | null>(null);
  const [tokenId, setTokenId] = useState<number | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const scene: EmbeddingScene = new EmbeddingScene(canvasRef.current, setTokenId);
    return () => scene.dispose();
  }, []);

  return (
    <>
      <canvas id="scene" ref={canvasRef} />
      <div id="hud">
        <div><b>embed_tokens</b> <span className="dim">[{VOCAB_SIZE}, {HIDDEN_SIZE}]</span> BF16 · 约 272 MB</div>
        <div>单个 Box 示意: 宽 {HIDDEN_SIZE} × 长 {VOCAB_SIZE}（游标 = 查表位置）</div>
        <div>当前查表 token id ≈ <span id="token">{tokenId === null ? '-' : tokenId}</span></div>
      </div>
      <div id="help">左键旋转 · 右键平移 · 滚轮缩放</div>
    </>
  );
}
