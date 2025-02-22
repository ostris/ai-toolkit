import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import SampleImageModal from '@/components/SampleImageModal';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Ostris - AI Toolkit',
  description: 'A toolkit for building AI things.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="apple-mobile-web-app-title" content="AI-Toolkit" />
      </head>
      <body className={inter.className}>
        <ThemeProvider>
          <div className="flex h-screen bg-gray-950">
            <Sidebar />
            <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative">{children}</main>
          </div>
        </ThemeProvider>
        <ConfirmModal />
        <SampleImageModal />
      </body>
    </html>
  );
}
