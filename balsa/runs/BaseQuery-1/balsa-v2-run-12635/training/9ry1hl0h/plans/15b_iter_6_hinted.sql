/*+ MergeJoin(k mi t cn mc ct at it1 mk)
 MergeJoin(mi t cn mc ct at it1 mk)
 HashJoin(mi t cn mc ct at it1)
 HashJoin(mi t cn mc ct at)
 NestLoop(t cn mc ct at)
 HashJoin(t cn mc ct)
 HashJoin(t cn mc)
 MergeJoin(cn mc)
 IndexScan(k)
 IndexScan(mi)
 SeqScan(t)
 IndexScan(cn)
 IndexScan(mc)
 SeqScan(ct)
 IndexScan(at)
 SeqScan(it1)
 IndexScan(mk)
 Leading((k (((mi (((t (cn mc)) ct) at)) it1) mk))) */
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS youtube_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND cn.name = 'YouTube'
  AND it1.info = 'release dates'
  AND mc.note LIKE '%(200%)%'
  AND mc.note LIKE '%(worldwide)%'
  AND mi.note LIKE '%internet%'
  AND mi.info LIKE 'USA:% 200%'
  AND t.production_year BETWEEN 2005 AND 2010
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

