/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        gray: {
          750: '#1f2937',
          850: '#111827',
          950: '#030712',
        },
      },
      animation: {
        'fadeIn': 'fadeIn 0.3s ease-in-out',
        'slideUp': 'slideUp 0.3s ease-in-out',
        'slideLeft': 'slideLeft 0.3s ease-in-out',
        'slideRight': 'slideRight 0.3s ease-in-out',
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce': 'bounce 1s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideLeft: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideRight: {
          '0%': { transform: 'translateX(-20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
      boxShadow: {
        soft: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      },
      typography: theme => ({
        DEFAULT: {
          css: {
            maxWidth: '100%',
            color: theme('colors.gray.800'),
            lineHeight: '1.6',
            p: {
              marginTop: '1em',
              marginBottom: '1em',
            },
            code: {
              color: theme('colors.gray.800'),
              backgroundColor: theme('colors.gray.100'),
              borderRadius: theme('borderRadius.md'),
              padding: '0.1em 0.3em',
              fontWeight: '500',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
            pre: {
              color: theme('colors.gray.200'),
              backgroundColor: theme('colors.gray.800'),
              borderRadius: theme('borderRadius.md'),
              padding: '1em',
              overflowX: 'auto',
            },
            'pre code': {
              backgroundColor: 'transparent',
              padding: '0',
              color: 'inherit',
            },
          },
        },
        invert: {
          css: {
            color: theme('colors.gray.200'),
            code: {
              color: theme('colors.gray.200'),
              backgroundColor: theme('colors.gray.800'),
            },
            pre: {
              color: theme('colors.gray.200'),
              backgroundColor: theme('colors.gray.900'),
            },
            h1: {
              color: theme('colors.gray.100'),
            },
            h2: {
              color: theme('colors.gray.100'),
            },
            h3: {
              color: theme('colors.gray.100'),
            },
            h4: {
              color: theme('colors.gray.100'),
            },
            strong: {
              color: theme('colors.gray.100'),
            },
            a: {
              color: theme('colors.primary.400'),
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} 