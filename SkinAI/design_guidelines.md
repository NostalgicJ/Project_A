# Design Guidelines: Cosmetics Ingredient Analysis Platform (코스메틱 성분 분석)

## Design Approach

**Reference-Based Hybrid Approach**: Drawing inspiration from premium beauty platforms (Sephora, Olive Young) combined with clean data-focused apps (Notion, Linear). The design emphasizes trust, cleanliness, and scientific credibility while maintaining Korean beauty industry aesthetics.

## Core Design Elements

### A. Color Palette

**Light Mode (Primary)**
- Background Base: 40 15% 97% (Warm off-white)
- Surface Cards: 35 20% 95% (Beige cream)
- Primary Brand: 25 60% 85% (Soft peach/apricot)
- Primary Hover: 25 65% 75%
- Text Primary: 20 15% 25% (Dark brown)
- Text Secondary: 25 10% 50% (Medium gray-brown)
- Border/Divider: 30 15% 88%
- Success (Safe): 140 45% 55% (Muted green)
- Warning (Caution): 35 70% 60% (Warm amber)
- Error (Harmful): 0 55% 65% (Soft coral red)
- Accent (Analysis): 200 45% 65% (Soft blue for AI features)

**Dark Mode**
- Background Base: 25 12% 12%
- Surface Cards: 25 15% 18%
- Primary Brand: 25 50% 65%
- Text Primary: 35 8% 92%
- Borders: 25 10% 25%

### B. Typography

**Primary**: Pretendard (Korean) / Inter (English) - via Google Fonts CDN
- Headings: 700 weight, tight letter-spacing
- Body: 400-500 weight, comfortable line-height (1.6)
- Small text (ingredients): 400 weight, slightly condensed

**Hierarchy**:
- H1 (Hero): 3.5rem (56px) / Mobile: 2.5rem
- H2 (Section): 2rem (32px) / Mobile: 1.75rem
- H3 (Card titles): 1.25rem (20px)
- Body: 1rem (16px)
- Small (metadata): 0.875rem (14px)

### C. Layout System

**Spacing Primitives**: Tailwind units of 2, 4, 6, 8, 12, 16, 20
- Component padding: p-6 (cards), p-8 (sections)
- Section spacing: py-16 md:py-20 (desktop), py-12 (mobile)
- Grid gaps: gap-6 for product cards, gap-4 for ingredient lists

**Container Widths**:
- Main content: max-w-6xl
- Product grids: Full width with max-w-7xl
- Forms/Analysis: max-w-2xl centered

### D. Component Library

**Navigation**
- Sticky header with blur backdrop (backdrop-blur-md)
- Logo left, main nav center, user profile/CTA right
- Mobile: Hamburger menu with slide-out drawer
- Include: 홈, 성분 분석, 내 화장대, 추천 제품

**Hero Section** (Landing Page)
- Full-width banner image: Clean, minimalist product photography on beige/cream background
- Centered content with semi-transparent overlay card
- Main CTA: "성분 분석 시작하기" (Start Ingredient Analysis)
- Secondary info: Trust indicators (AI 분석, 전문가 검증, 안전성 평가)

**Product Cards** (내 화장대)
- Rounded corners (rounded-xl)
- Product image top (aspect-ratio-square or 4:3)
- Product name, brand (smaller text)
- Safety score badge (colored pill with icon)
- Ingredient count indicator
- Hover: Subtle lift (shadow-lg) and scale (scale-105)
- Grid: 2 columns mobile, 3-4 columns desktop

**Analysis Display** (성분 분석 결과)
- Two-column layout: Left (ingredient list with color-coded safety), Right (AI analysis summary)
- Ingredient items: Pills/badges with safety color coding
- Expandable details for each ingredient (click to expand)
- Overall safety score: Large circular progress indicator
- Compatibility warnings: Alert-style cards with icons

**Input/Forms**
- Product name input with autocomplete suggestions
- Ingredient paste area: Large textarea with placeholder
- Image upload: Drag-and-drop zone with preview
- Submit button: Prominent with loading state animation

**Data Displays**
- Safety ratings: Color-coded progress bars
- Ingredient categories: Grouped accordion lists
- Compatibility matrix: Visual grid/table showing product interactions
- Recommendations: Card carousel with swipe navigation

### E. Component Enrichment

**Header**: Logo + Navigation + Search bar + User menu + "분석 시작" CTA button

**Footer**: Newsletter signup ("성분 정보 뉴스레터 구독"), Quick links (서비스 소개, 성분 사전, 고객센터), Social media, Trust badges (인증마크), Contact info

**Landing Page Sections**:
1. Hero with background image
2. Feature highlights (3-column: AI 분석, 성분 사전, 맞춤 추천)
3. How it works (4-step process with icons)
4. Sample analysis showcase (before/after style)
5. User testimonials (2-3 column cards with photos)
6. CTA section with trial signup form

**Product Collection Page** (내 화장대):
- Filter/sort toolbar
- Grid view/list view toggle
- Empty state with illustration and CTA
- Bulk actions toolbar when items selected

## Images Strategy

**Hero Section**: Yes - Full-width lifestyle image showing clean cosmetics arrangement on minimalist beige surface (1920x800px)

**Additional Images**:
- Product cards: Square product photos (400x400px)
- Feature section icons: Use Heroicons for simplicity
- Testimonials: Circular user photos (80x80px)
- How-it-works: Illustrated icons or minimal graphics
- Empty states: Custom illustration of cosmetics bottle with magnifying glass

## Key UX Principles

- Immediate value: Show sample analysis on homepage without requiring login
- Progressive disclosure: Ingredient details expand on click, not shown by default
- Trust building: Display safety scores prominently with clear explanations
- Mobile-first analysis: Ensure ingredient lists are scannable on small screens
- Loading states: Skeleton screens during AI analysis (3-5 seconds)
- Microinteractions: Subtle animations on card hover, badge pulse for warnings

## Accessibility Notes

- Maintain WCAG AA contrast ratios throughout
- Color-blind safe: Use icons + text for safety ratings, not color alone
- Dark mode: Full support with consistent theming
- Form inputs: Visible focus states with accent color rings
- Korean font rendering: Ensure proper weight rendering for Hangul characters