# راهنمای فارسی بازبینی نامزدهای قطع و بازخورد کوتاه

این مرحله **ضبط صدا نیست**. فقط ۲۴ برش کوتاه از صوت موجود و مجاز برای آموزش داخلی شنیده می‌شود؛ مجموع زمان شنیدن برش‌ها ۱۶۸٫۳ ثانیه است. برگه از هر یک از چهار کانال Digiato، Mehran Rowshan Persian، Tabaghe16 و Zoomit شش نمونه دارد.

این بازبینی با فایل ۴۰ ردیفی `conversation_manual_qa.csv` فرق دارد:

- فایل ۴۰ ردیفی، کیفیت تعداد/انتساب گوینده، کپشن، هم‌ترازی و نویز پنجره‌ها را می‌سنجد؛
- فایل ۲۴ ردیفی `conversation_interruption_qa.csv` فقط مرز دقیق دو نوبت و شنیدنی‌بودن هم‌پوشانی را می‌سنجد.

## شنیدن هر ردیف

از ریشه پروژه، شماره ردیف داده را از ۱ تا ۲۴ اجرا کنید:

```bash
.venv/bin/python scripts/review_interaction_candidate.py --row 1
```

ابزار فقط همان بازه `listen_start_s` با طول `listen_duration_s` را با `ffplay` پخش می‌کند؛ فایل صوتی جدیدی نمی‌سازد. برای دیدن متن و فرمان بدون پخش:

```bash
.venv/bin/python scripts/review_interaction_candidate.py --row 1 --show-only
```

اگر `ffplay` موجود نبود، همان `audio_filepath` را در هر پخش‌کننده باز کنید و فقط بازه `listen_start_s` تا `listen_end_s` را بشنوید. استاد، عضو آزمایشگاه یا برچسب‌زن مورد تأیید نیز می‌تواند این کار را انجام دهد.

## پرکردن ستون‌ها

برای هر ردیف در `data/processed/manifests/conversation_interruption_qa.csv`:

- `review_status`: اگر نامزد معتبر است `pass` و اگر نیست `fail`؛
- `speakers_distinct_correct`: آیا واقعاً دو گوینده متفاوت‌اند؟
- `user_turn_boundary_correct`: آیا مرز پایان نوبت گوینده اول درست است؟
- `response_turn_boundary_correct`: آیا مرز شروع گوینده دوم درست است؟
- `audible_overlap_correct`: آیا هم‌پوشانی واقعاً شنیده می‌شود؟
- `corrected_label`: برای ردیف قبول‌شده فقط `interrupt` یا `backchannel`؛
- `reviewer_id`: شناسه غیرحساس بازبین؛
- `notes`: توضیح اختیاری.

برای `pass` باید هر چهار ستون صحت `yes` باشند. اگر شک دارید، `fail` بزنید و علت را در `notes` بنویسید. مقدار `automatic_candidate` فقط پیشنهاد الگوریتم است و نباید بدون شنیدن کپی شود.

نمونه تعریف عملی:

- `interrupt`: گوینده دوم پیش از پایان نوبت گوینده اول شروع می‌کند و نوبت مستقل/ادامه‌دار می‌گیرد؛
- `backchannel`: پاسخ بسیار کوتاه مانند تأیید یا همراهی است و گفتار گوینده اول عملاً ادامه دارد.

## اعمال نتیجه به‌صورت fail-closed

پس از تکمیل هر دو برگه QA:

```bash
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl

.venv/bin/python -m thesis_s2s.cli apply-interruption-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
```

فقط ردیف کامل و قبول‌شده به برچسب انسانی تبدیل می‌شود. ردیف ردشده، ناقص، ناشناخته یا بررسی‌نشده همچنان صرفاً نامزد خودکار می‌ماند. ستون‌های شواهد زمان‌بندی در CSV دوباره از مانیفست محاسبه می‌شوند؛ دست‌کاری آن‌ها روی خروجی اثر ندارد. فرمان نمونه‌گیری نیز اگر پاسخ بازبین در CSV وجود داشته باشد از بازنویسی فایل خودداری می‌کند.

اگر دقت نمونه مورد قبول استاد نبود، نامزدها نباید گروهی تأیید شوند؛ در آن حالت باید تعداد بیشتری بازبینی شود یا مکمل/اصلاح دامنه مورد تأیید استاد استفاده گردد.
