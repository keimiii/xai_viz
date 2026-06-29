/**
 * Main App component with routing.
 */

import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HomePage } from './pages/Home';
import { ImageDetailPage } from './pages/ImageDetail';
import { ComparePage } from './pages/Compare';
import { DashboardPage } from './pages/Dashboard';
import { Q2Page } from './pages/Q2';
import { Q3ReportPage } from './pages/Q3Report';
import { NotFoundPage } from './pages/NotFound';
import { RouteErrorBoundary } from './components/ui/ErrorBoundary';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

function AppShell() {
  const location = useLocation();
  const shellWidthClass = location.pathname.startsWith('/image/')
    ? 'max-w-[96rem]'
    : location.pathname === '/q3-report'
      ? 'max-w-[110rem]'
      : 'max-w-7xl';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className={`${shellWidthClass} mx-auto px-4 sm:px-6 lg:px-8`}>
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <NavLink to="/" className="text-xl font-bold text-gray-900">
                SSL Attention
              </NavLink>
            </div>

            <div className="flex items-center space-x-4">
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md ${
                    isActive
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`
                }
              >
                Gallery
              </NavLink>
              <NavLink
                to="/compare"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md ${
                    isActive
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`
                }
              >
                Compare
              </NavLink>
              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md ${
                    isActive
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`
                }
              >
                Dashboard
              </NavLink>
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className={`${shellWidthClass} mx-auto px-4 sm:px-6 lg:px-8 py-8`}>
        <Routes>
          <Route path="/" element={<RouteErrorBoundary><HomePage /></RouteErrorBoundary>} />
          <Route path="/image/:imageId" element={<RouteErrorBoundary><ImageDetailPage /></RouteErrorBoundary>} />
          <Route path="/compare" element={<RouteErrorBoundary><ComparePage /></RouteErrorBoundary>} />
          <Route path="/dashboard" element={<RouteErrorBoundary><DashboardPage /></RouteErrorBoundary>} />
          <Route path="/q2" element={<RouteErrorBoundary><Q2Page /></RouteErrorBoundary>} />
          <Route path="/q3-report" element={<RouteErrorBoundary><Q3ReportPage /></RouteErrorBoundary>} />
          <Route path="*" element={<RouteErrorBoundary><NotFoundPage /></RouteErrorBoundary>} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8">
        <div className={`${shellWidthClass} mx-auto px-4 sm:px-6 lg:px-8 py-4`}>
          <p className="text-center text-sm text-gray-500">
            SSL Attention Visualization - WikiChurches Dataset
          </p>
        </div>
      </footer>
    </div>
  );
}
