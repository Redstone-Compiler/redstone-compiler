use std::collections::HashMap;
use std::fs;
use std::path::Path;

use eyre::Result;

/// 선형 계획법 제약 조건의 부등호 방향
#[derive(Clone, Copy, Debug)]
pub enum Sense {
    /// <= (작거나 같음)
    Le,
    /// >= (크거나 같음)
    Ge,
    /// = (같음)
    Eq,
}

/// 선형 계획법 제약 조건
#[derive(Clone, Debug)]
pub struct Constraint {
    pub name: String,
    pub terms: Vec<(f64, usize)>, // (계수, 변수 인덱스)
    pub sense: Sense,
    pub rhs: f64, // 우변 값
}

/// 선형 계획법 모델
#[derive(Default)]
pub struct Lp {
    /// 변수 이름 목록
    pub var_names: Vec<String>,
    /// 변수 이름 -> 인덱스 매핑
    pub var_index: HashMap<String, usize>,
    /// 각 변수가 이진 변수인지 여부
    pub is_binary: Vec<bool>,
    /// 목적 함수 항들 (계수, 변수 인덱스)
    pub objective: Vec<(f64, usize)>,
    /// 제약 조건 목록
    pub constraints: Vec<Constraint>,
    /// 최소화 여부 (true면 최소화, false면 최대화)
    pub minimize: bool,
}

impl Lp {
    /// 새로운 LP 모델 생성
    pub fn new() -> Self {
        Self {
            minimize: true,
            ..Default::default()
        }
    }

    /// 이진 변수 추가 (또는 기존 변수 반환)
    pub fn bin(&mut self, name: String) -> usize {
        if let Some(&i) = self.var_index.get(&name) {
            return i;
        }
        let i = self.var_names.len();
        self.var_names.push(name.clone());
        self.var_index.insert(name, i);
        self.is_binary.push(true);
        i
    }

    /// 목적 함수에 항 추가
    pub fn add_obj(&mut self, coef: f64, var: usize) {
        self.objective.push((coef, var));
    }

    /// 제약 조건 추가
    pub fn add_c(&mut self, name: String, terms: Vec<(f64, usize)>, sense: Sense, rhs: f64) {
        self.constraints.push(Constraint {
            name,
            terms,
            sense,
            rhs,
        });
    }

    /// LP 파일 형식으로 저장 (CPLEX LP 형식)
    pub fn write_lp(&self, path: &Path) -> Result<()> {
        let mut s = String::new();

        // 목적 함수
        if self.minimize {
            s.push_str("Minimize\n obj: ");
        } else {
            s.push_str("Maximize\n obj: ");
        }

        if self.objective.is_empty() {
            s.push_str("0\n");
        } else {
            s.push_str(&format_linexpr(&self.objective, &self.var_names));
            s.push('\n');
        }

        // 제약 조건
        s.push_str("Subject To\n");
        for c in &self.constraints {
            s.push_str(" ");
            s.push_str(&c.name);
            s.push_str(": ");
            s.push_str(&format_linexpr(&c.terms, &self.var_names));
            match c.sense {
                Sense::Le => s.push_str(" <= "),
                Sense::Ge => s.push_str(" >= "),
                Sense::Eq => s.push_str(" = "),
            }
            s.push_str(&format!("{}", c.rhs));
            s.push('\n');
        }

        // 변수 범위
        s.push_str("Bounds\n");
        for (i, name) in self.var_names.iter().enumerate() {
            if self.is_binary[i] {
                s.push_str(&format!(" 0 <= {} <= 1\n", name));
            } else {
                s.push_str(&format!(" {} free\n", name));
            }
        }

        // 이진 변수 선언
        s.push_str("Binaries\n");
        for (i, name) in self.var_names.iter().enumerate() {
            if self.is_binary[i] {
                s.push_str(" ");
                s.push_str(name);
                s.push('\n');
            }
        }
        s.push_str("End\n");

        fs::write(path, s)?;
        Ok(())
    }
}

/// 선형 표현식을 문자열로 포맷팅
fn format_linexpr(terms: &[(f64, usize)], var_names: &[String]) -> String {
    let mut out = String::new();
    let mut first = true;
    for &(coef, vid) in terms {
        if coef == 0.0 {
            continue;
        }
        if first {
            first = false;
            out.push_str(&format_term(coef, &var_names[vid], true));
        } else {
            out.push_str(&format_term(coef, &var_names[vid], false));
        }
    }
    if out.is_empty() {
        out.push('0');
    }
    out
}

/// 단일 항을 문자열로 포맷팅
fn format_term(coef: f64, name: &str, first: bool) -> String {
    let sign = if coef < 0.0 { "-" } else { "+" };
    let abs = coef.abs();
    if first {
        if coef < 0.0 {
            if abs == 1.0 {
                format!("- {}", name)
            } else {
                format!("- {} {}", abs, name)
            }
        } else if abs == 1.0 {
            format!("{}", name)
        } else {
            format!("{} {}", abs, name)
        }
    } else if abs == 1.0 {
        format!(" {} {}", sign, name)
    } else {
        format!(" {} {} {}", sign, abs, name)
    }
}
