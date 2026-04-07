'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from './ThemeProvider';

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  // Button styled as the opposite theme so it stands out
  const buttonClass =
    theme === 'dark'
      ? 'bg-[rgb(215,215,215)] hover:bg-[rgb(195,195,195)] text-[rgb(82,82,82)]'
      : 'bg-[rgb(23,23,23)] hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)]';

  return (
    <button
      onClick={toggleTheme}
      className={`flex items-center justify-center p-1 rounded-lg transition-colors ${buttonClass}`}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
    </button>
  );
};

export default ThemeToggle;
