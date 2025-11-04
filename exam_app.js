// DBMLA 試験問題 - アプリケーションロジック（改善版）

let allQuestions = []; // 全ての問題セットを統合した配列
let selectedChoices = {};
let currentQuestionId = null;
let questionTags = {}; // 問題ごとのタグを保存

// 全問題セットをロードして統合
function loadAllQuestions() {
    allQuestions = [];

    // 過去問1
    if (window.questions) {
        window.questions.forEach(q => {
            allQuestions.push({...q, examSet: 1, uniqueId: `1-${q.number}`});
        });
    }

    // 過去問2
    if (window.questions2) {
        window.questions2.forEach(q => {
            allQuestions.push({...q, examSet: 2, uniqueId: `2-${q.number}`});
        });
    }

    // 過去問3
    if (window.questions3) {
        window.questions3.forEach(q => {
            allQuestions.push({...q, examSet: 3, uniqueId: `3-${q.number}`});
        });
    }
}

// HTMLエスケープ関数
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// 解説内のコードブロックをPrism.js形式に変換
function convertCodeBlocks(html) {
    return html.replace(/<div class="code-block">([\s\S]*?)<\/div>/g, (match, code) => {
        return `<pre><code class="language-python">${code}</code></pre>`;
    });
}

// 初期化
document.addEventListener('DOMContentLoaded', () => {
    // 全問題セットをロード
    loadAllQuestions();

    loadFromLocalStorage(); // ローカルストレージから読み込み
    populateDomainCheckboxes(); // ドメインチェックボックスを生成
    renderQuestionsList();
    setupKeyboardShortcuts();
    updateFilterCount(); // フィルタカウント更新
    initializeSections(); // セクションの初期状態を設定
    showFirstQuestion(); // 最初の問題を表示
});

// セクションの初期化（問題一覧のみ展開）
function initializeSections() {
    // 問題一覧のみ展開、他は折りたたみ
    toggleSection('questions', true); // 強制展開
}

// 最初の問題を表示
function showFirstQuestion() {
    const filteredQuestions = filterQuestions();
    if (filteredQuestions.length > 0) {
        showQuestionDetail(filteredQuestions[0].uniqueId);
    }
}

// セクションの展開/折りたたみ
function toggleSection(sectionName, forceExpand = false) {
    const content = document.getElementById(`${sectionName}-content`);
    const icon = document.getElementById(`${sectionName}-icon`);

    if (!content || !icon) return;

    const isExpanded = content.classList.contains('expanded');

    if (forceExpand || !isExpanded) {
        // 展開
        content.classList.add('expanded');
        icon.textContent = '▼';
    } else {
        // 折りたたみ
        content.classList.remove('expanded');
        icon.textContent = '▲';
    }
}

// フィルタパネルの開閉（後方互換性のため残す）
function toggleFilterPanel() {
    toggleSection('filter');
}

// フィルタサブセクションの展開/折りたたみ
function toggleFilterSubsection(subsectionName) {
    const content = document.getElementById(`${subsectionName}-content`);
    const icon = document.getElementById(`${subsectionName}-icon`);

    if (!content || !icon) return;

    const isExpanded = content.classList.contains('expanded');

    if (isExpanded) {
        // 折りたたみ
        content.classList.remove('expanded');
        icon.textContent = '▲';
    } else {
        // 展開
        content.classList.add('expanded');
        icon.textContent = '▼';
    }
}

// すべてのチェックボックスをON/OFF
function toggleAllCheckboxes(checkboxName) {
    const checkboxes = document.querySelectorAll(`input[name="${checkboxName}"]`);

    // 現在の状態を確認（すべてチェック済みならOFF、それ以外ならON）
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);

    checkboxes.forEach(cb => {
        cb.checked = !allChecked;
    });
}

// ドメインチェックボックスを動的生成
function populateDomainCheckboxes() {
    const domainSet = new Set();
    allQuestions.forEach(q => {
        if (q.domain) {
            domainSet.add(q.domain);
        }
    });

    const domainCheckboxes = document.getElementById('domain-checkboxes');
    const domains = Array.from(domainSet).sort();

    domainCheckboxes.innerHTML = domains.map(domain => `
        <label>
            <input type="checkbox" name="domain" value="${domain}" checked>
            ${domain}
        </label>
    `).join('');
}

// フィルタを適用
function applyFilters() {
    renderQuestionsList();
    updateFilterCount();
}

// フィルタカウント更新
function updateFilterCount() {
    const filterCount = document.getElementById('filter-count');

    // アクティブなフィルタ数をカウント
    let activeFilters = 0;

    // 問題セット
    const examSetChecks = document.querySelectorAll('input[name="examSet"]:checked');
    if (examSetChecks.length < 3) {
        activeFilters++;
    }

    // ステータス
    const statusRadio = document.querySelector('input[name="status"]:checked');
    if (statusRadio && statusRadio.value !== 'all') {
        activeFilters++;
    }

    // ドメイン
    const allDomains = document.querySelectorAll('input[name="domain"]');
    const checkedDomains = document.querySelectorAll('input[name="domain"]:checked');
    if (checkedDomains.length < allDomains.length) {
        activeFilters++;
    }

    // タグ
    const tagRadio = document.querySelector('input[name="tag"]:checked');
    if (tagRadio && tagRadio.value !== 'all') {
        activeFilters++;
    }

    filterCount.textContent = activeFilters > 0 ? `(${activeFilters})` : '';
}

// ローカルストレージから読み込み（全セット統合版）
function loadFromLocalStorage() {
    // 各問題セットのデータを読み込み
    for (let set = 1; set <= 3; set++) {
        const storageKey = `dbmla_progress_set${set}`;
        const saved = localStorage.getItem(storageKey);
        if (saved) {
            const data = JSON.parse(saved);
            if (data.selectedChoices) {
                Object.keys(data.selectedChoices).forEach(num => {
                    selectedChoices[`${set}-${num}`] = data.selectedChoices[num];
                });
            }
            if (data.questionTags) {
                Object.keys(data.questionTags).forEach(num => {
                    questionTags[`${set}-${num}`] = data.questionTags[num];
                });
            }
        }
    }
}

// ローカルストレージに保存（全セット統合版）
function saveToLocalStorage() {
    // 各問題セットごとにデータを分離して保存
    for (let set = 1; set <= 3; set++) {
        const storageKey = `dbmla_progress_set${set}`;
        const setChoices = {};
        const setTags = {};

        Object.keys(selectedChoices).forEach(key => {
            if (key.startsWith(`${set}-`)) {
                const num = key.split('-')[1];
                setChoices[num] = selectedChoices[key];
            }
        });

        Object.keys(questionTags).forEach(key => {
            if (key.startsWith(`${set}-`)) {
                const num = key.split('-')[1];
                setTags[num] = questionTags[key];
            }
        });

        const data = {
            selectedChoices: setChoices,
            questionTags: setTags
        };
        localStorage.setItem(storageKey, JSON.stringify(data));
    }
}

// キーボードショートカットの設定（uniqueId対応）
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // 入力欄にフォーカスがある場合は無視
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        let filteredQuestions = filterQuestions();
        const currentIndex = filteredQuestions.findIndex(q => q.uniqueId === currentQuestionId);

        switch(e.key) {
            case 'ArrowRight':
            case 'n':
            case 'N':
                // 次の問題
                e.preventDefault();
                if (currentQuestionId && currentIndex < filteredQuestions.length - 1) {
                    showQuestionDetail(filteredQuestions[currentIndex + 1].uniqueId);
                } else if (!currentQuestionId && filteredQuestions.length > 0) {
                    showQuestionDetail(filteredQuestions[0].uniqueId);
                }
                break;

            case 'ArrowLeft':
            case 'p':
            case 'P':
                // 前の問題
                e.preventDefault();
                if (currentQuestionId && currentIndex > 0) {
                    showQuestionDetail(filteredQuestions[currentIndex - 1].uniqueId);
                }
                break;

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
                // 選択肢を選択 (1-5)
                e.preventDefault();
                if (currentQuestionId) {
                    const choiceIndex = parseInt(e.key) - 1;
                    const question = allQuestions.find(q => q.uniqueId === currentQuestionId);
                    if (question && choiceIndex < question.choices.length) {
                        selectChoice(currentQuestionId, choiceIndex);
                    }
                }
                break;

            case 'a':
            case 'A':
            case 'b':
            case 'B':
            case 'c':
            case 'C':
            case 'd':
            case 'D':
            case 'e':
            case 'E':
                // 選択肢を選択 (A-E)
                e.preventDefault();
                if (currentQuestionId) {
                    const choiceIndex = e.key.toUpperCase().charCodeAt(0) - 65;
                    const question = allQuestions.find(q => q.uniqueId === currentQuestionId);
                    if (question && choiceIndex < question.choices.length) {
                        selectChoice(currentQuestionId, choiceIndex);
                    }
                }
                break;

            case 'r':
            case 'R':
                // リセット
                e.preventDefault();
                resetAll();
                break;

            case 't':
            case 'T':
                // タグのオンオフを切り替え
                e.preventDefault();
                if (currentQuestionId) {
                    toggleTag(currentQuestionId);
                }
                break;

            case 'l':
            case 'L':
                // 問題リストのトグル（モバイル）
                e.preventDefault();
                toggleQuestionsList();
                break;

            case '?':
                // ヘルプ表示
                e.preventDefault();
                showShortcutHelp();
                break;
        }
    });
}

// ショートカットヘルプを表示
function showShortcutHelp() {
    const helpText = `
📌 キーボードショートカット

▶ ナビゲーション:
  • N / → : 次の問題
  • P / ← : 前の問題
  • L : 問題リストを開く/閉じる

▶ 回答:
  • A-E : 選択肢を選択
  • 1-5 : 選択肢を選択

▶ その他:
  • T : タグのオンオフを切り替え
  • R : リセット
  • ? : このヘルプを表示
    `;
    alert(helpText);
}

// 問題リストを描画（複数フィルタ対応）
function renderQuestionsList() {
    const container = document.getElementById('questions-cards');
    if (!container) {
        console.error('questions-cards element not found');
        return;
    }

    // フィルタリング
    let filteredQuestions = filterQuestions();

    container.innerHTML = '';

    filteredQuestions.forEach(q => {
        const card = document.createElement('div');
        card.className = 'question-card';

        const isAnswered = selectedChoices[q.uniqueId] !== undefined;
        const isCorrect = isAnswered && selectedChoices[q.uniqueId] === q.correctIndex;

        if (isAnswered) {
            card.classList.add('answered');
            if (isCorrect) {
                card.classList.add('correct');
            } else {
                card.classList.add('incorrect');
            }
        }
        if (currentQuestionId === q.uniqueId) {
            card.classList.add('active');
        }

        card.onclick = () => showQuestionDetail(q.uniqueId);

        // カードタグ
        const cardTags = document.createElement('div');
        cardTags.className = 'card-tags';
        if (questionTags[q.uniqueId]) {
            const tagIndicator = document.createElement('span');
            tagIndicator.className = 'tag-indicator';
            tagIndicator.textContent = '⭐';
            cardTags.appendChild(tagIndicator);
        }

        card.innerHTML = `
            <div class="card-header">
                <span class="card-number">問題 ${q.examSet}-${q.number}</span>
                <span class="card-domain">${escapeHtml(q.domain)}</span>
            </div>
            <div class="card-question">${escapeHtml(q.question)}</div>
        `;
        card.appendChild(cardTags);

        container.appendChild(card);
    });

    // 統計を更新
    updateStats();
}

// フィルタリング処理
function filterQuestions() {
    // 新しいチェックボックスベースのUIから取得
    const examSetChecks = document.querySelectorAll('input[name="examSet"]:checked');
    const statusRadio = document.querySelector('input[name="status"]:checked');
    const domainChecks = document.querySelectorAll('input[name="domain"]:checked');
    const tagRadio = document.querySelector('input[name="tag"]:checked');

    // 選択された問題セットを取得
    const selectedExamSets = Array.from(examSetChecks).map(cb => parseInt(cb.value));

    // 選択されたドメインを取得
    const selectedDomains = Array.from(domainChecks).map(cb => cb.value);

    // ステータスとタグの値を取得
    const statusFilter = statusRadio ? statusRadio.value : 'all';
    const tagFilter = tagRadio ? tagRadio.value : 'all';

    let filteredQuestions = allQuestions;

    // 問題セットフィルタ
    if (selectedExamSets.length > 0) {
        filteredQuestions = filteredQuestions.filter(q => selectedExamSets.includes(q.examSet));
    }

    // ドメインフィルタ
    if (selectedDomains.length > 0) {
        filteredQuestions = filteredQuestions.filter(q => selectedDomains.includes(q.domain));
    }

    // ステータスフィルタ
    if (statusFilter === 'unanswered') {
        filteredQuestions = filteredQuestions.filter(q => selectedChoices[q.uniqueId] === undefined);
    } else if (statusFilter === 'correct') {
        filteredQuestions = filteredQuestions.filter(q =>
            selectedChoices[q.uniqueId] !== undefined &&
            selectedChoices[q.uniqueId] === q.correctIndex
        );
    } else if (statusFilter === 'incorrect') {
        filteredQuestions = filteredQuestions.filter(q =>
            selectedChoices[q.uniqueId] !== undefined &&
            selectedChoices[q.uniqueId] !== q.correctIndex
        );
    }

    // タグフィルタ
    if (tagFilter === 'tagged') {
        filteredQuestions = filteredQuestions.filter(q => questionTags[q.uniqueId]);
    } else if (tagFilter === 'untagged') {
        filteredQuestions = filteredQuestions.filter(q => !questionTags[q.uniqueId]);
    }

    return filteredQuestions;
}

// 問題詳細を表示（uniqueId対応）
function showQuestionDetail(uniqueId) {
    currentQuestionId = uniqueId;
    const question = allQuestions.find(q => q.uniqueId === uniqueId);
    if (!question) return;

    const container = document.getElementById('question-detail');

    // ナビゲーションボタンの状態を決定
    let filteredQuestions = filterQuestions();
    const currentIndex = filteredQuestions.findIndex(q => q.uniqueId === uniqueId);
    const hasPrev = currentIndex > 0;
    const hasNext = currentIndex < filteredQuestions.length - 1;
    const prevQuestion = hasPrev ? filteredQuestions[currentIndex - 1].uniqueId : null;
    const nextQuestion = hasNext ? filteredQuestions[currentIndex + 1].uniqueId : null;

    let choicesHTML = question.choices.map((choice, index) => {
        const letter = String.fromCharCode(65 + index);
        const isSelected = selectedChoices[uniqueId] === index;
        const isCorrect = index === question.correctIndex;
        const showResult = selectedChoices[uniqueId] !== undefined;

        let choiceClass = 'choice';
        if (isSelected) choiceClass += ' selected';
        if (showResult && isCorrect) choiceClass += ' correct';
        if (showResult && isSelected && !isCorrect) choiceClass += ' incorrect';

        return `
            <div class="${choiceClass}" onclick="selectChoice('${uniqueId}', ${index})">
                <span class="choice-label">${letter}.</span>
                <span class="choice-text">${escapeHtml(choice)}</span>
            </div>
        `;
    }).join('');

    const codeBlock = question.code ? `<pre><code class="language-python">${escapeHtml(question.code)}</code></pre>` : '';

    const showAnswer = selectedChoices[uniqueId] !== undefined;

    // タグアイコンボタン（右上配置）
    const hasTag = questionTags[uniqueId] || false;
    const tagButton = `<button class="tag-icon-btn ${hasTag ? 'active' : ''}" onclick="toggleTag('${uniqueId}')" title="${hasTag ? 'タグを外す' : 'タグを付ける'}">★</button>`;

    container.innerHTML = `
        <div class="detail-content">
            <div class="question-header">
                <div class="header-left">
                    <div class="question-number">問題 ${question.examSet}-${question.number}</div>
                    <div class="domain-badge">${escapeHtml(question.domain)}</div>
                </div>
                ${tagButton}
            </div>
            <div class="question-text">${escapeHtml(question.question)}</div>
            ${codeBlock}
            <div class="choices">${choicesHTML}</div>
            ${showAnswer ? `
                <div class="answer-section">
                    <div class="answer-content show">
                        <div class="key-point">
                            <strong>📌 要点:</strong> ${escapeHtml(question.keyPoint)}
                        </div>
                        <div class="correct-answer">
                            正解: ${String.fromCharCode(65 + question.correctIndex)}. ${escapeHtml(question.choices[question.correctIndex])}
                        </div>
                        <div class="explanation">
                            ${convertCodeBlocks(question.explanation)}
                        </div>
                    </div>
                </div>
            ` : ''}
            <div class="copy-section">
                <button class="copy-question-btn" onclick="copyQuestion('${uniqueId}', event)">📋 問題をコピー</button>
            </div>
        </div>
    `;

    // 固定の次へボタンを更新
    updateFixedNextButton(hasNext, nextQuestion);

    // Prism.jsでシンタックスハイライトを適用
    if (typeof Prism !== 'undefined') {
        setTimeout(() => {
            Prism.highlightAll();
        }, 0);
    }

    renderQuestionsList(); // リストを更新してアクティブ状態を反映
    hideQuestionsList(); // モバイルで問題を選択したらリストを閉じる
}

// 固定の次へボタンを更新
function updateFixedNextButton(hasNext, nextQuestion) {
    let fixedBtn = document.getElementById('fixed-next-btn');

    if (!fixedBtn) {
        // ボタンが存在しない場合は作成
        fixedBtn = document.createElement('button');
        fixedBtn.id = 'fixed-next-btn';
        fixedBtn.className = 'fixed-next-btn';
        document.body.appendChild(fixedBtn);
    }

    if (hasNext) {
        fixedBtn.disabled = false;
        fixedBtn.onclick = () => showQuestionDetail(nextQuestion);
        fixedBtn.innerHTML = '次の問題 <span style="font-size: 1.3em;">→</span>';
    } else {
        fixedBtn.disabled = true;
        fixedBtn.onclick = null;
        fixedBtn.innerHTML = '最後の問題 <span style="font-size: 1.3em;">✓</span>';
    }
}

// 選択肢を選択（uniqueId対応）
function selectChoice(uniqueId, choiceIndex) {
    selectedChoices[uniqueId] = choiceIndex;
    saveToLocalStorage();
    showQuestionDetail(uniqueId);

    // 解説部分まで自動スクロール
    setTimeout(() => {
        const answerSection = document.querySelector('.answer-section');
        if (answerSection) {
            answerSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }, 100);
}

// 問題をクリップボードにコピー（uniqueId対応）
function copyQuestion(uniqueId, event) {
    const question = allQuestions.find(q => q.uniqueId === uniqueId);
    if (!question) return;

    // コピー用テキストを生成
    let text = `問題 ${question.examSet}-${question.number} (${question.domain})\n\n`;
    text += `【問題文】\n${question.question}\n\n`;

    if (question.code) {
        text += `【コード】\n${question.code}\n\n`;
    }

    text += `【選択肢】\n`;
    question.choices.forEach((choice, index) => {
        const letter = String.fromCharCode(65 + index);
        text += `${letter}. ${choice}\n`;
    });

    text += `\n【正解】\n${String.fromCharCode(65 + question.correctIndex)}. ${question.choices[question.correctIndex]}\n\n`;
    text += `【解説】\n${question.explanation}\n`;

    // クリップボードにコピー
    navigator.clipboard.writeText(text).then(() => {
        // コピー成功のフィードバック
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '✓ コピーしました！';
        btn.style.background = '#4caf50';

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.style.background = '';
        }, 2000);
    }).catch(err => {
        alert('コピーに失敗しました: ' + err);
    });
}

// タグをトグル（uniqueId対応）
function toggleTag(uniqueId) {
    const currentState = questionTags[uniqueId] || false;
    questionTags[uniqueId] = !currentState;

    // falseの場合は削除
    if (!questionTags[uniqueId]) {
        delete questionTags[uniqueId];
    }

    saveToLocalStorage();
    renderQuestionsList();
    showQuestionDetail(uniqueId);
}

// 問題リストの表示/非表示を切り替え（モバイル用）
function toggleQuestionsList() {
    const list = document.getElementById('questions-list');
    list.classList.toggle('show');
}

// 問題リストを非表示（モバイル用）
function hideQuestionsList() {
    if (window.innerWidth <= 768) {
        const list = document.getElementById('questions-list');
        list.classList.remove('show');
    }
}

// フィルタをリセット
function resetFilters() {
    // 問題セット: すべて選択
    document.querySelectorAll('input[name="examSet"]').forEach(cb => {
        cb.checked = true;
    });

    // ステータス: すべて
    const statusAll = document.querySelector('input[name="status"][value="all"]');
    if (statusAll) statusAll.checked = true;

    // ドメイン: すべて選択
    document.querySelectorAll('input[name="domain"]').forEach(cb => {
        cb.checked = true;
    });

    // タグ: すべて
    const tagAll = document.querySelector('input[name="tag"][value="all"]');
    if (tagAll) tagAll.checked = true;

    renderQuestionsList();
    updateFilterCount();
}

// すべてリセット
function resetAll() {
    if (!confirm('すべての回答履歴とタグをリセットしますか？')) {
        return;
    }

    selectedChoices = {};
    questionTags = {};
    currentQuestionId = null;

    // フィルタもリセット
    resetFilters();

    const container = document.getElementById('question-detail');
    container.innerHTML = '<div class="detail-empty">← 左側から問題を選択してください</div>';

    // LocalStorageもクリア
    localStorage.clear();

    renderQuestionsList();
}

// 苦手対策シート作成（uniqueId対応）
function createCheatSheet() {
    // タグ付き問題または間違えた問題を取得
    const targetQuestions = allQuestions.filter(q => {
        const hasTag = questionTags[q.uniqueId];
        const isIncorrect = selectedChoices[q.uniqueId] !== undefined &&
                           selectedChoices[q.uniqueId] !== q.correctIndex;
        return hasTag || isIncorrect;
    });

    if (targetQuestions.length === 0) {
        alert('対象となる問題がありません。\n苦手な問題に★タグを付けるか、問題を解いてから実行してください。');
        return;
    }

    // ドメインごとにグループ化
    const byDomain = {};
    targetQuestions.forEach(q => {
        if (!byDomain[q.domain]) {
            byDomain[q.domain] = [];
        }

        // タグと不正解の情報を追加
        const hasTag = questionTags[q.uniqueId];
        const isIncorrect = selectedChoices[q.uniqueId] !== undefined &&
                           selectedChoices[q.uniqueId] !== q.correctIndex;

        byDomain[q.domain].push({
            question: q,
            hasTag: hasTag,
            isIncorrect: isIncorrect
        });
    });

    // テキストファイルの内容を生成
    let content = '=' .repeat(60) + '\n';
    content += '　　　　DBMLA 苦手対策シート\n';
    content += '　　　Databricks Machine Learning Associate\n';
    content += '=' .repeat(60) + '\n\n';
    content += `作成日時: ${new Date().toLocaleString('ja-JP')}\n`;
    content += `対象問題数: ${targetQuestions.length}問\n`;

    const taggedCount = targetQuestions.filter(q => questionTags[q.uniqueId]).length;
    const incorrectCount = targetQuestions.filter(q =>
        selectedChoices[q.uniqueId] !== undefined &&
        selectedChoices[q.uniqueId] !== q.correctIndex
    ).length;
    content += `　- ★タグ付き: ${taggedCount}問\n`;
    content += `　- 不正解: ${incorrectCount}問\n\n`;

    // ドメインごとに出力
    Object.keys(byDomain).sort().forEach(domain => {
        content += '\n' + '─'.repeat(60) + '\n';
        content += `【${domain}】\n`;
        content += '─'.repeat(60) + '\n\n';

        byDomain[domain].forEach(item => {
            const q = item.question;
            const markers = [];
            if (item.hasTag) markers.push('★');
            if (item.isIncorrect) markers.push('×');
            const markerText = markers.length > 0 ? `[${markers.join(' ')}] ` : '';

            content += `■ 問題 ${q.examSet}-${q.number} ${markerText}\n`;
            content += `　${q.keyPoint}\n\n`;
        });
    });

    content += '\n' + '='.repeat(60) + '\n';
    content += '凡例:\n';
    content += '　★ = タグ付き問題（要復習マーク）\n';
    content += '　× = 不正解だった問題\n';
    content += '　★ × = タグ付き かつ 不正解\n\n';
    content += '復習のポイント:\n';
    content += '- 各要点を理解し、自分の言葉で説明できるか確認しましょう\n';
    content += '- 実際のコードを書いて動作を確認することをお勧めします\n';
    content += '- 関連する公式ドキュメントも合わせて確認しましょう\n';
    content += '='.repeat(60) + '\n';

    // ファイルとしてダウンロード
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DBMLA_cheatsheet_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    alert(`苦手対策シートを作成しました！\n\n対象問題数: ${targetQuestions.length}問\n　- ★タグ付き: ${taggedCount}問\n　- 不正解: ${incorrectCount}問`);
}

// 統計を更新（uniqueId対応）
function updateStats() {
    const filteredQuestions = filterQuestions();
    document.getElementById('visible-count').textContent = filteredQuestions.length;

    const answeredCount = Object.keys(selectedChoices).length;
    const totalCount = allQuestions.length;
    document.getElementById('answered-count').textContent = `${answeredCount}/${totalCount}`;

    // 正答率を計算
    if (answeredCount === 0) {
        document.getElementById('accuracy-rate').textContent = '-%';
    } else {
        let correctCount = 0;
        for (let uniqueId in selectedChoices) {
            const question = allQuestions.find(q => q.uniqueId === uniqueId);
            if (question && selectedChoices[uniqueId] === question.correctIndex) {
                correctCount++;
            }
        }
        const accuracy = Math.round((correctCount / answeredCount) * 100);
        document.getElementById('accuracy-rate').textContent = `${accuracy}%`;
    }
}
