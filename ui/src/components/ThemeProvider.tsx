'use client';

import { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext({ isDark: true });

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [isDark, setIsDark] = useState(true);

  return <ThemeContext.Provider value={{ isDark }}>{children}</ThemeContext.Provider>;
};
