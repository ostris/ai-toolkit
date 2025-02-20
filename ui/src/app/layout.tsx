import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Ostris - AI Toolkit',
  description: 'A toolkit for building AI things.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <ThemeProvider>
          <div className="flex h-screen bg-gray-950">
            <Sidebar />
            <main className="flex-1 p-8 overflow-auto bg-gray-950 text-gray-100">{children}</main>
          </div>
        </ThemeProvider>
        <ConfirmModal />
      </body>
    </html>
  );
}
