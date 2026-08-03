'use client';

import { useTheme } from './ThemeProvider';

const ThemeLogo = () => {
  const { theme } = useTheme();
  const src = theme === 'dark' ? '/ostris_logo.png' : '/ostris_logo_black.png';

  return <img src={src} alt="Ostris AI Toolkit" className="w-auto h-7 mr-3 inline" />;
};

export default ThemeLogo;
