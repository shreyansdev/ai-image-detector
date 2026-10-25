import React from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/Home';
import DetectorPage from './pages/Detector';
import HowItWorksPage from './pages/HowItWorks';
import AboutPage from './pages/About';
import FAQPage from './pages/FAQ';
import NotFoundPage from './pages/NotFound';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: '/',
        element: <HomePage />,
      },
      {
        path: '/detector',
        element: <DetectorPage />,
      },
      {
        path: '/how-it-works',
        element: <HowItWorksPage />,
      },
      {
        path: '/about',
        element: <AboutPage />,
      },
      {
        path: '/faq',
        element: <FAQPage />,
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
]);

export const App: React.FC = () => {
  return <RouterProvider router={router} />;
};

export default App;
