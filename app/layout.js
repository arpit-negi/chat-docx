import './globals.css';

export const metadata = {
  title: 'Chat with Your Document',
  description: 'Upload a document and ask it questions using AI',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">{children}</body>
    </html>
  );
}
