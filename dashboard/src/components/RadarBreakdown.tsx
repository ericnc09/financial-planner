import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

interface Props {
  signalScore: number;
  fundamentalScore: number;
  macroModifier: number;
  conviction: number;
}

export const RadarBreakdown: React.FC<Props> = ({ signalScore, fundamentalScore, macroModifier, conviction }) => {
  const data = [
    { factor: 'Signal', value: signalScore * 100 },
    { factor: 'Fundamental', value: fundamentalScore * 100 },
    { factor: 'Macro', value: ((macroModifier - 0.5) / 1.0) * 100 },
    { factor: 'Conviction', value: conviction * 100 },
  ];

  return (
    <div style={{ width: '100%', height: 250 }}>
      <ResponsiveContainer>
        <RadarChart data={data}>
          <PolarGrid stroke="#30363d" />
          <PolarAngleAxis dataKey="factor" tick={{ fill: '#8b949e', fontSize: 11 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
